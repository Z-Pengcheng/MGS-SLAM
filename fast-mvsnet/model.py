import logging

import pytorch_lightning as pl
import timm
import torch
from torch import Tensor
import torch.nn.functional as F
from .modules.losses import MSGradientLoss, MVDepthLoss, NormalsLoss, ScaleInvariantLoss
from .modules.cost_volume import DotCostVolumeManager
from .modules.networks import CVEncoder, DepthDecoderPP, CostRegularization, MatchingEncoder
from torch import nn
from .utils.geometry_utils import NormalGenerator
from .utils.metrics_utils import compute_depth_metrics

logger = logging.getLogger(__name__)

class FastMVNet(pl.LightningModule):
    """
    FastMVNet
    """
    def __init__(self, opts):
        super().__init__()
        self.save_hyperparameters()

        self.run_opts = opts

        # Coarse to fine levels
        self.levels = len(self.run_opts.depth_num)

        # depth_num
        self.depth_nums = self.run_opts.depth_num

        # iniitalize the encoder for strong image priors
        self.encoder = timm.create_model(
                                        "tf_efficientnetv2_s_in21ft1k", 
                                        pretrained=False, 
                                        features_only=True
                                    )

        self.encoder.num_ch_enc = self.encoder.feature_info.channels()

        # init the matching encoder. 
        # level-0, level-1, level-2
        self.matching_model = MatchingEncoder()

        # what type of cost volume are we using?
        self.cost_volume = DotCostVolumeManager()

        # iniitalize the first half of the U-Net, encoding the cost volume 
        # and image prior image feautres
        self.cost_volume_net = CVEncoder(
            num_ch_enc=self.encoder.num_ch_enc,
            num_ch_outs=[self.depth_nums[0], self.depth_nums[1], self.depth_nums[2],
                         self.depth_nums[2]*4, self.depth_nums[2]*8]
        )

        # iniitalize the final depth decoder
        if self.run_opts.depth_decoder_name == "unet_pp":
            self.depth_decoder = DepthDecoderPP(self.cost_volume_net.num_ch_enc)
        elif self.run_opts.depth_decoder_name == "cost_reg":
            self.depth_decoder = CostRegularization(self.cost_volume_net.num_ch_enc, self.depth_nums)

        # all the losses
        if self.run_opts.loss_type == "log_l1":
            self.ms_loss_fn = nn.L1Loss()
        elif self.run_opts.loss_type == "smooth_l1":
            self.ms_loss_fn = nn.SmoothL1Loss(reduction='mean')

        self.si_loss = ScaleInvariantLoss()
        self.grad_loss = MSGradientLoss()
        self.normals_loss = NormalsLoss()
        self.mv_depth_loss = MVDepthLoss(
                                self.run_opts.image_height // 2,
                                self.run_opts.image_width // 2,
                            )

        # used for normals loss
        self.compute_normals = NormalGenerator(
                                    self.run_opts.image_height // 2, 
                                    self.run_opts.image_width // 2,
                                )

    def compute_matching_feats(
                            self, 
                            cur_image, 
                            src_image,
                            flip,
                        ):
        """ 
            Computes matching features for the current image (reference) and 
            source images.
        """
        # Compute matching features and batch them to reduce variance from 
        # batchnorm when training.
        images = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)
        matching_feats = self.matching_model(images)

        matching_cur_feats = {} 
        matching_src_feats = {}

        for level in matching_feats:
            feats_l = matching_feats[level]
            feats_l = feats_l.view(B, V, *feats_l.shape[1:])
            matching_cur_feats[level] = feats_l[:, 0]
            matching_src_feats[level] = feats_l[:, 1:].contiguous()

            if flip:
                # now (carefully) flip matching features back for correct MVS.
                matching_cur_feats[level] = torch.flip(matching_cur_feats[level], (-1,))
                matching_src_feats[level] = torch.flip(matching_src_feats[level], (-1,))

        return matching_cur_feats, matching_src_feats
    
    def generate_depth_planes(self,
                              batch_size: int, 
                              num_depth: int,
                              min_depth: float,
                              max_depth: float,
                              matching_height: int,
                              matching_width: int) -> Tensor:
        """ 
            generate depth planes
        """

        linear_ramp_1d11 = torch.linspace(min_depth,
                                          max_depth,
                                          num_depth).view(1, num_depth, 1, 1).cuda()

        depth_planes_bdhw = linear_ramp_1d11.expand(batch_size, 
                                                    num_depth, 
                                                    matching_height, 
                                                    matching_width)

        return depth_planes_bdhw

    def predict_depth(self,
                      level: int,
                      prior_feats: Tensor,
                      cur_feats: Tensor,
                      src_feats: Tensor,
                      src_extrinsics: Tensor,
                      src_Ks: Tensor,
                      cur_invK: Tensor,
                      depth_planes_bdhw: Tensor,
                      flip: bool):
        
        cost_volume = self.cost_volume(
                                      cur_feats=cur_feats,
                                      src_feats=src_feats,
                                      src_extrinsics=src_extrinsics,
                                      src_Ks=src_Ks,
                                      cur_invK=cur_invK,
                                      depth_planes_bdhw = depth_planes_bdhw
                                    )
        
        if flip:
            # OK, we've computed the cost volume, now we need to flip the cost
            # volume to have it aligned with flipped image prior features
            cost_volume = torch.flip(cost_volume, (-1,))

        # Encode the cost volume and current image features 
        cost_volume_features = self.cost_volume_net(
                                                level,
                                                cost_volume, 
                                                prior_feats,
                                            )
        # prior_feats = prior_feats[:level] + cost_volume_features

        # Decode into depth at multiple resolutions.
        depth_outputs = self.depth_decoder(level, cost_volume_features)

        if self.run_opts.depth_decoder_name == "unet_pp":
            # loop through depth outputs, flip them if we need to and get linear 
            # scale depths.
            for k in list(depth_outputs.keys()):
                log_depth = depth_outputs[k].float()

                if flip:
                    # now flip the depth map back after final prediction
                    log_depth = torch.flip(log_depth, (-1,))

                depth_outputs[k] = log_depth
                depth_outputs[k.replace("log_", "")] = torch.exp(log_depth)

        elif self.run_opts.depth_decoder_name == "cost_reg":
            for k in list(depth_outputs.keys()):
                pb_depth = depth_outputs[k].float()

                if flip:
                    # now flip the depth map back after final prediction
                    pb_depth = torch.flip(pb_depth, (-1,))

                pb_depth = F.softmax(pb_depth, 1)

                depth_outputs[k] = torch.sum(
                                        pb_depth * 
                                        depth_planes_bdhw, 
                                        dim=1,
                                        keepdim=True
                                    )

        return depth_outputs

    def forward(
            self,
            phase,
            cur_data,
            src_data
        ):
        """ 
            Computes a forward pass through the depth model.
        """
        # get all tensors from the batch dictioanries.
        cur_image = cur_data["image_b3hw"]
        src_image = src_data["image_b3hw"]
        src_cam_T_world = src_data["cam_T_world_b44"]
        cur_world_T_cam = cur_data["world_T_cam_b44"]

        with torch.cuda.amp.autocast(False):
            # Compute src_cam_T_cur_cam, a transformation for going from 3D 
            # coords in current view coordinate frame to source view coords 
            # coordinate frames.
            src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

        # flip transformation! Figure out if we're flipping. Should be true if
        # we are training and a coin flip says we should.
        flip_threshold = 0.5 if phase == "train" else 0.0
        flip = torch.rand(1).item() < flip_threshold

        if flip:
            # flip all images.
            cur_image = torch.flip(cur_image, (-1,))
            src_image = torch.flip(src_image, (-1,))

        # Compute image features for the current view. Used for a strong image
        # prior.
        prior_feats = self.encoder(cur_image)

        # Compute matching features
        # (B*V, 8, H//2, W//2), (B*V, 16, H//4, W//4), (B*V, 32, H//8, W//8)
        matching_cur_feats, matching_src_feats = self.compute_matching_feats(cur_image, src_image, flip)

        # depth outputs of 3 stages 
        depth_outputs = {}
        # Coarse to fine 3 levels
        for i in reversed(range(self.levels)):
            src_K = src_data[f"K_l{i}_b44"]
            cur_invK = cur_data[f"invK_l{i}_b44"]
            depth_num = self.depth_nums[i]
            interval_ratio = self.run_opts.interval_ratios[i]
            matching_cur_feat = matching_cur_feats[f"level_{i}"]
            matching_src_feat = matching_src_feats[f"level_{i}"]

            B, _, h, w = matching_cur_feat.shape

            # stage 1
            if i == (self.levels-1):
                # Get min and max depth to the right shape, device and dtype
                min_depth = self.run_opts.min_matching_depth
                max_depth = self.run_opts.max_matching_depth
                depth_planes_bdhw = self.generate_depth_planes(B, depth_num, min_depth, max_depth, h, w)

                # predict depth
                depth_output = self.predict_depth(level=i,
                                                  prior_feats = prior_feats,
                                                  cur_feats=matching_cur_feat,
                                                  src_feats=matching_src_feat,
                                                  src_extrinsics=src_cam_T_cur_cam,
                                                  src_Ks=src_K,
                                                  cur_invK=cur_invK,
                                                  depth_planes_bdhw=depth_planes_bdhw,
                                                  flip=flip)
                depth_outputs.update(depth_output)
            
            # stage 2, 3
            else:
                # Get min and max depth to the right shape, device and dtype
                min_depth = -(self.run_opts.depth_interval / interval_ratio) * (depth_num / 2)
                max_depth = (self.run_opts.depth_interval / interval_ratio) * (depth_num / 2)
                depth_planes_bdhw = self.generate_depth_planes(B, depth_num, min_depth, max_depth, h, w)

                # get last depth bdhw
                last_depth_b1hw = depth_output[f"depth_pred_l{i+1}_b1hw"].detach()
                last_depth_b1hw = F.interpolate(last_depth_b1hw,
                                                scale_factor=2, mode='bilinear',
                                                align_corners=True)
                last_depth_bdhw = last_depth_b1hw.repeat_interleave(depth_num, dim=1)
                depth_planes_bdhw = last_depth_bdhw + depth_planes_bdhw

                # predict depth
                depth_output = self.predict_depth(level=i,
                                                  prior_feats = prior_feats,
                                                  cur_feats=matching_cur_feat,
                                                  src_feats=matching_src_feat,
                                                  src_extrinsics=src_cam_T_cur_cam,
                                                  src_Ks=src_K,
                                                  cur_invK=cur_invK,
                                                  depth_planes_bdhw=depth_planes_bdhw,
                                                  flip=flip)
                
                depth_outputs.update(depth_output)

        return depth_outputs

    def compute_losses(self, cur_data, src_data, outputs):
        """ 
        Compute losses for the current view's depth.
        """
        depth_gt = cur_data["depth_b1hw"]
        normals_gt = cur_data["normals_b3hw"]
        mask_b = cur_data["mask_b_b1hw"]
        depth_pred = outputs["depth_pred_l0_b1hw"]
        normals_pred = outputs["normals_pred_b3hw"]

        found_scale = False
        ms_loss = 0
        si_loss = 0
        if self.run_opts.depth_decoder_name == "unet_pp":
            log_depth_gt = torch.log(depth_gt)
            for i in range(3):
                if f"log_depth_pred_l{i}_b1hw" in outputs:
                    log_depth_pred_resized = F.interpolate(
                                            outputs[f"log_depth_pred_l{i}_b1hw"], 
                                            size=depth_gt.shape[-2:],
                                            mode="nearest",
                                        )
                    ms_loss += self.ms_loss_fn(
                                                log_depth_gt[mask_b], 
                                                log_depth_pred_resized[mask_b]
                                            ) / 2 ** i
                    found_scale = True
        elif self.run_opts.depth_decoder_name == "cost_reg":
            log_depth_gt = torch.log(depth_gt)
            for i in range(3):
                if f"depth_pred_l{i}_b1hw" in outputs:
                    depth_pred_resized = F.interpolate(
                                                outputs[f"depth_pred_l{i}_b1hw"], 
                                                size=depth_gt.shape[-2:],
                                                mode="nearest",
                                            )
                    log_depth_pred = torch.log(depth_pred_resized)
                    si_loss += self.si_loss(
                                            log_depth_gt[mask_b],
                                            log_depth_pred[mask_b]
                                           ) / 2 ** i
                    found_scale = True
        if not found_scale:
            raise Exception("Could not find a valid scale to compute si loss!")

        grad_loss = self.grad_loss(depth_gt, depth_pred)

        normals_loss = self.normals_loss(normals_gt, normals_pred)

        mv_loss = self.mv_depth_loss(
                            depth_pred_b1hw=depth_pred,
                            cur_depth_b1hw=depth_gt,
                            src_depth_bk1hw=src_data["depth_b1hw"],
                            cur_invK_b44=cur_data[f"invK_l0_b44"],
                            src_K_bk44=src_data[f"K_l0_b44"],
                            cur_world_T_cam_b44=cur_data["world_T_cam_b44"],
                            src_cam_T_world_bk44=src_data["cam_T_world_b44"],
                        )

        loss = si_loss + 1.0 * grad_loss + 1.0 * normals_loss + 0.2 * mv_loss

        losses = {
            "loss": loss,
            "si_loss": si_loss,
            "grad_loss": grad_loss,
            "normals_loss": normals_loss,
            "mv_loss": mv_loss,
        }
        return losses

    def step(self, phase, batch, batch_idx):
        """ Takes a training/validation step through the model.
        
            phase: "train" or "val". "train" will signal this function and 
                others log results and use flip augmentation.
            batch: (cur_data, src_data) where cur_data is a dict with data on 
                the current (reference) view and src_data is a dict with data on
                source views. 
        """
        cur_data, src_data = batch

        # forward pass through the model.
        outputs = self(phase, cur_data, src_data)

        
        depth_pred = outputs["depth_pred_l0_b1hw"]

        depth_gt = cur_data["depth_b1hw"]
        mask_b = cur_data["mask_b_b1hw"]

        # estimate normals for groundtruth
        normals_gt = self.compute_normals(depth_gt, cur_data["invK_l0_b44"])
        cur_data["normals_b3hw"] = normals_gt

        # estimate normals for depth
        normals_pred = self.compute_normals(depth_pred, cur_data["invK_l0_b44"])
        outputs["normals_pred_b3hw"] = normals_pred

        # compute losses
        losses = self.compute_losses(cur_data, src_data, outputs)
        
        # is train?
        is_train = phase == "train"

        # logging and validation
        with torch.inference_mode():
            # log losses
            for loss_name, loss_val in losses.items():
                self.log(f'{phase}/{loss_name}', 
                        loss_val, 
                        sync_dist=True, 
                        on_step=is_train, 
                        on_epoch=not is_train)

            # validation
            metrics = compute_depth_metrics(
                                        depth_gt[mask_b], depth_pred[mask_b])

            # validation log
            for metric_name, metric_val in metrics.items():
                self.log(f'{phase}_metrics/{metric_name}', 
                            metric_val, 
                            sync_dist=True, 
                            on_step=is_train, 
                            on_epoch=not is_train)

        return losses["loss"]

    def training_step(self, batch, batch_idx):
        """ Runs a training step. """
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """ Runs a validation step. """
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        """ Configuring optmizers and learning rate schedules. 
            
            By default we use a stepped learning rate schedule with steps at 
            70000 and 80000.

        """
        optimizer = torch.optim.AdamW(self.parameters(), 
                            lr=self.run_opts.lr, weight_decay=self.run_opts.wd)
        def lr_lambda(step):
            if step < self.run_opts.lr_steps[0]:
                return 1
            elif step < self.run_opts.lr_steps[1]:
                return .1
            else:
                return .01
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}
