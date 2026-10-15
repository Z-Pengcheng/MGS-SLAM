import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.slam_utils import get_median_depth, optimize_depth
from utils.depth_predict import mvs_network

from dpvo import fastba
from dpvo import altcorr
from dpvo.net import VONet
from dpvo.lietorch import SE3
from dpvo.utils import flatmeshgrid
from dpvo import projective_ops as pops

import open3d as o3d
import matplotlib.pyplot as plt
import cv2

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.map_kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.load_vo_weight(self.config["sparse_vo"]["vo_weight_path"])
        self.prior_depth_network = mvs_network(self.config)

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

        self.vo_n = 0
        self.vo_m = 0
        self.vo_M = self.config["sparse_vo"]["patches_per_frame"]
        self.vo_N = self.config["sparse_vo"]["buffer_size"]
        self.vo_ht = self.dataset.height
        self.vo_wd = self.dataset.width
        self.tstamps_ = dict()
        self.vo_poses_ = torch.zeros(self.vo_N, 7, dtype=torch.float, device=self.device)
        self.vo_patches_ = torch.zeros(self.vo_N, self.vo_M, 3, self.vo_network.P, self.vo_network.P, dtype=torch.float, device=self.device)
        self.vo_intrinsics_ = torch.zeros(self.vo_N, 4, dtype=torch.float, device=self.device)
        self.vo_index_ = torch.zeros(self.vo_N, self.vo_M, dtype=torch.long, device=self.device)

        self.vo_mem = 32
        self.vo_kwargs = kwargs = {"device": self.device, "dtype": torch.half}
        self.vo_imap_ = torch.zeros(self.vo_mem, self.vo_M, self.vo_network.DIM, **kwargs)
        self.vo_gmap_ = torch.zeros(self.vo_mem, self.vo_M, 128, self.vo_network.P, self.vo_network.P, **kwargs)

        ht = self.dataset.height // self.vo_network.RES
        wd = self.dataset.width // self.vo_network.RES
        self.vo_fmap1_ = torch.zeros(1, self.vo_mem, 128, ht // 1, wd // 1, **kwargs)
        self.vo_fmap2_ = torch.zeros(1, self.vo_mem, 128, ht // 4, wd // 4, **kwargs)
        self.vo_pyramid = (self.vo_fmap1_, self.vo_fmap2_)

        self.vo_net = torch.zeros(1, 0, self.vo_network.DIM, **kwargs)
        self.vo_ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.vo_jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.vo_kk = torch.as_tensor([], dtype=torch.long, device=self.device)
        
        self.vo_poses_[:,6] = 1.0
        self.vo_removal_window = self.config["sparse_vo"]["removal_window"]
        self.vo_optimization_window = self.config["sparse_vo"]["optimization_window"]
        self.vo_patch_lifetime = self.config["sparse_vo"]["patch_lifetime"]
        self.vo_keyframe_index = self.config["sparse_vo"]["keyframe_index"]
        self.vo_keyframe_thresh = self.config["sparse_vo"]["keyframe_thresh"]
        self.vo_initial_depth = self.config["sparse_vo"]["vo_initial_depth"]

    @property
    def vo_poses(self):
        return self.vo_poses_.view(1, self.vo_N, 7)
    @property
    def vo_patches(self):
        return self.vo_patches_.view(1, self.vo_N*self.vo_M, 3, 3, 3)
    @property
    def vo_intrinsics(self):
        return self.vo_intrinsics_.view(1, self.vo_N, 4)
    @property
    def vo_ix(self):
        return self.vo_index_.view(-1)
    @property
    def vo_imap(self):
        return self.vo_imap_.view(1, self.vo_mem * self.vo_M, self.vo_network.DIM)
    @property
    def vo_gmap(self):
        return self.vo_gmap_.view(1, self.vo_mem * self.vo_M, 128, 3, 3)

    def load_vo_weight(self, vo_weight_path):
        from collections import OrderedDict
        state_dict = torch.load(vo_weight_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "update.lmbda" not in k:
                new_state_dict[k.replace('module.', '')] = v
        
        self.vo_network = VONet()
        self.vo_network.load_state_dict(new_state_dict)
        self.vo_network.cuda()
        self.vo_network.eval()

    def corr(self, coords, indicies=None):
        ii, jj = indicies if indicies is not None else (self.vo_kk, self.vo_jj)
        ii1 = ii % (self.vo_M * self.vo_mem)
        jj1 = jj % (self.vo_mem)
        corr1 = altcorr.corr(self.vo_gmap, self.vo_pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.vo_gmap, self.vo_pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        (ii, jj, kk) = indicies if indicies is not None else (self.vo_ii, self.vo_jj, self.vo_kk)
        coords = pops.transform(SE3(self.vo_poses), self.vo_patches, self.vo_intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.vo_jj = torch.cat([self.vo_jj, jj])
        self.vo_kk = torch.cat([self.vo_kk, ii])
        self.vo_ii = torch.cat([self.vo_ii, self.vo_ix[ii]])

        net = torch.zeros(1, len(ii), self.vo_network.DIM, **self.vo_kwargs)
        self.vo_net = torch.cat([self.vo_net, net], dim=1)

    def remove_factors(self, m):
        self.vo_ii = self.vo_ii[~m]
        self.vo_jj = self.vo_jj[~m]
        self.vo_kk = self.vo_kk[~m]
        self.vo_net = self.vo_net[:,~m]

    def motion_probe(self):
        kk = torch.arange(self.vo_m-self.vo_M, self.vo_m, device=self.device)
        jj = self.vo_n * torch.ones_like(kk)
        ii = self.vo_ix[kk]

        net = torch.zeros(1, len(ii), self.vo_network.DIM, **self.vo_kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with torch.cuda.amp.autocast(enabled=True):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.vo_imap[:,kk % (self.vo_M * self.vo_mem)]
            net, (delta, weight, _) = \
                self.vo_network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.vo_ii == i) & (self.vo_jj == j)
        ii = self.vo_ii[k]
        jj = self.vo_jj[k]
        kk = self.vo_kk[k]

        flow = pops.flow_mag(SE3(self.vo_poses), self.vo_patches, self.vo_intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def vo_keyframe(self):
        create_kf_vo = False

        i = self.vo_n - self.vo_keyframe_index - 1
        j = self.vo_n - self.vo_keyframe_index + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
        if m / 2 < self.vo_keyframe_thresh:
            k = self.vo_n - self.vo_keyframe_index
            to_remove = (self.vo_ii == k) | (self.vo_jj == k)
            self.remove_factors(to_remove)

            self.vo_kk[self.vo_ii > k] -= self.vo_M
            self.vo_ii[self.vo_ii > k] -= 1
            self.vo_jj[self.vo_jj > k] -= 1

            for i in range(k, self.vo_n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.vo_poses_[i] = self.vo_poses_[i+1]
                self.vo_patches_[i] = self.vo_patches_[i+1]
                self.vo_intrinsics_[i] = self.vo_intrinsics_[i+1]

                self.vo_imap_[i%self.vo_mem] = self.vo_imap_[(i+1) % self.vo_mem]
                self.vo_gmap_[i%self.vo_mem] = self.vo_gmap_[(i+1) % self.vo_mem]
                self.vo_fmap1_[0,i%self.vo_mem] = self.vo_fmap1_[0,(i+1)%self.vo_mem]
                self.vo_fmap2_[0,i%self.vo_mem] = self.vo_fmap2_[0,(i+1)%self.vo_mem]

            self.vo_n -= 1
            self.vo_m-= self.vo_M
            to_remove = self.vo_ix[self.vo_kk] < self.vo_n - self.vo_removal_window
            self.remove_factors(to_remove)
        else:
            create_kf_vo = True

        return create_kf_vo

    def vo_update(self):
        coords = self.reproject()

        with torch.cuda.amp.autocast(enabled=True):
            corr = self.corr(coords)
            ctx = self.vo_imap[:,self.vo_kk % (self.vo_M * self.vo_mem)]
            self.vo_net, (delta, weight, _) = \
                self.vo_network.update(self.vo_net, ctx, corr, None, self.vo_ii, self.vo_jj, self.vo_kk)

        lmbda = torch.as_tensor([1e-4], device=self.device)
        weight = weight.float()
        target = coords[...,self.vo_network.P//2,self.vo_network.P//2] + delta.float()
        t0 = self.vo_n - self.vo_optimization_window
        t0 = max(t0, 1)
        try:
            fastba.BA(self.vo_poses, self.vo_patches, self.vo_intrinsics, 
                target, weight, lmbda, self.vo_ii, self.vo_jj, self.vo_kk, t0, self.vo_n, 2)
        except:
            Log("Warning VO BA failed.")

    def __edges_forw(self):
        r=self.vo_patch_lifetime
        t0 = self.vo_M * max((self.vo_n - r), 0)
        t1 = self.vo_M * max((self.vo_n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(self.vo_n-1, self.vo_n, device=self.device), indexing='ij')

    def __edges_back(self):
        r=self.vo_patch_lifetime
        t0 = self.vo_M * max((self.vo_n - 1), 0)
        t1 = self.vo_M * max((self.vo_n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device=self.device),
            torch.arange(max(self.vo_n-r, 0), self.vo_n, device=self.device), indexing='ij')

    def add_map_keyframe(self, key_frame_idx, depth=None, opacity=None):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.map_kf_indices.append(key_frame_idx)
        viewpoint = self.cameras[key_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if depth == None:
            # use the observed depth
            initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
            initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        else:
            depth = depth.detach().clone()
            opacity = opacity.detach()
            
            median_depth, std, valid_mask = get_median_depth(
                depth, opacity, mask=valid_rgb, return_std=True
            )
            invalid_depth_mask = torch.logical_or(
                depth > median_depth + std, depth < median_depth - std
            )
            invalid_depth_mask = torch.logical_or(
                invalid_depth_mask, ~valid_mask
            )
            depth[invalid_depth_mask] = median_depth
            initial_depth = depth + torch.randn_like(depth) * torch.where(
                invalid_depth_mask, std * 0.5, std * 0.2
            )

            initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
        return initial_depth.cpu().numpy()[0]


    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = True
        self.map_kf_indices = []
        self.occ_aware_visibility = {}
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        self.tstamps_[self.vo_n] = cur_frame_idx
            
        with torch.no_grad():
            vo_image = self.cameras[cur_frame_idx].original_image
            vo_image = 2 * vo_image[None, None] - 0.5
            with torch.cuda.amp.autocast(enabled=True):
                fmap, gmap, imap, patches, _, = \
                    self.vo_network.patchify(vo_image, self.vo_M)
            
            self.vo_intrinsics_[self.vo_n] = \
                torch.Tensor([
                    self.cameras[cur_frame_idx].fx,
                    self.cameras[cur_frame_idx].fy,
                    self.cameras[cur_frame_idx].cx,
                    self.cameras[cur_frame_idx].cy
                ]) / self.vo_network.RES
            
            self.vo_index_[self.vo_n + 1] = self.vo_n + 1

            if self.vo_n > 1:
                P1 = SE3(self.vo_poses_[self.vo_n-1])
                P2 = SE3(self.vo_poses_[self.vo_n-2])
                xi = 0.5 * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.vo_poses_[self.vo_n] = tvec_qvec

            vo_initial_depth = self.vo_initial_depth * torch.ones_like(patches[:,:,2,0,0,None,None])
            patches[:,:,2] = 1 / (vo_initial_depth + torch.rand_like(patches[:,:,2,0,0,None,None]))
            # patches[:,:,2] = torch.ones_like(patches[:,:,2,0,0,None,None])
            self.vo_patches_[self.vo_n] = patches
            self.vo_imap_[self.vo_n % self.vo_mem] = imap.squeeze()
            self.vo_gmap_[self.vo_n % self.vo_mem] = gmap.squeeze()
            self.vo_fmap1_[:, self.vo_n % self.vo_mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.vo_fmap2_[:, self.vo_n % self.vo_mem] = F.avg_pool2d(fmap[0], 4, 4)

            if self.vo_n > 0:
                if self.motion_probe() < 2.0:
                    self.cleanup(cur_frame_idx)
                    return 

            self.vo_n += 1
            self.vo_m += self.vo_M

            self.append_factors(*self.__edges_forw())
            self.append_factors(*self.__edges_back())

            if self.vo_n == 8:
                self.reset=False
                for itr in range(12):
                    self.vo_update()

                for itr in range(self.vo_n):
                    index = self.tstamps_[itr]
                    pose = SE3(self.vo_poses_[itr]).matrix().detach().clone()
                    self.cameras[index].update_RT(pose[0:3, 0:3].clone(), pose[0:3, 3].clone())

                index = [self.tstamps_[itr] for itr in [0, 1, 2, 3, 4, 5, 6, 7]]
                prior_dense_depth = self.prior_depth_network.get_depth([self.cameras[itr] for itr in index]).cpu().numpy()[0]
                sparse_depth = self.get_sparse_depth(index[0])
                opt_dense_depth = optimize_depth(sparse_depth, prior_dense_depth)
                self.cameras[index[0]].depth = opt_dense_depth
                depth_map = self.add_map_keyframe(index[0])
                self.current_window.append(index[0])
                self.request_init(index[0], self.cameras[index[0]], depth_map)

    def get_sparse_depth(self, vo_keyframe_id):
        poses = SE3(self.vo_poses)
        patches = self.vo_patches[:, :self.vo_m-self.vo_keyframe_index*self.vo_M]
        intrinsics = self.vo_intrinsics
        ix_all = self.vo_ix[:self.vo_m-self.vo_keyframe_index*self.vo_M]
        ix_target = torch.ones_like(ix_all)
        ix_target = self.vo_ix[ix_target*(self.vo_m-self.vo_keyframe_index*self.vo_M)]
        intrinsic = o3d.core.Tensor([[self.cameras[vo_keyframe_id].fx, 0, self.cameras[vo_keyframe_id].cx], 
                                     [0, self.cameras[vo_keyframe_id].fy, self.cameras[vo_keyframe_id].cy],
                                     [0, 0, 1]])

        sparse_points = pops.point_cloud(poses, patches, intrinsics, ix_all)
        sparse_points = poses[:, ix_target, None, None] * sparse_points
        sparse_points = (sparse_points[...,1,1,:3] / sparse_points[...,1,1,3:]).reshape(-1, 3)
        sparse_points = sparse_points.cpu().numpy()

        sparse_pcd = o3d.t.geometry.PointCloud(sparse_points)
        sparse_depth = sparse_pcd.project_to_depth_image(
            width=self.cameras[vo_keyframe_id].image_width,
            height=self.cameras[vo_keyframe_id].image_height,
            intrinsics=intrinsic,
            depth_scale=1.0,
            depth_max=10.0
        )

        sparse_depth = sparse_depth.to_legacy()
        sparse_depth = np.array(sparse_depth)

        return sparse_depth

    def vo_tracking(self, cur_frame_idx):
        self.tstamps_[self.vo_n] = cur_frame_idx

        with torch.no_grad():
            vo_image = self.cameras[cur_frame_idx].original_image
            vo_image = 2 * vo_image[None, None] - 0.5
            with torch.cuda.amp.autocast(enabled=True):
                fmap, gmap, imap, patches, _, = \
                    self.vo_network.patchify(vo_image, self.vo_M)
                
            self.vo_intrinsics_[self.vo_n] = \
                torch.Tensor([
                    self.cameras[cur_frame_idx].fx,
                    self.cameras[cur_frame_idx].fy,
                    self.cameras[cur_frame_idx].cx,
                    self.cameras[cur_frame_idx].cy
                ]) / self.vo_network.RES
            
            self.vo_index_[self.vo_n + 1] = self.vo_n + 1

            if self.vo_n > 1:
                P1 = SE3(self.vo_poses_[self.vo_n-1])
                P2 = SE3(self.vo_poses_[self.vo_n-2])
                xi = 0.5 * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.vo_poses_[self.vo_n] = tvec_qvec

            init_pose = SE3(self.vo_poses_[self.vo_n]).matrix().clone().detach()
            self.cameras[cur_frame_idx].R = init_pose[0:3, 0:3]
            self.cameras[cur_frame_idx].T = init_pose[0:3, 3]
            render_pkg = render(
                    self.cameras[cur_frame_idx], self.gaussians, self.pipeline_params, self.background)
            rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
            gt_img = self.cameras[cur_frame_idx].original_image.cuda()
            valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

            # get depth median
            vo_initial_depth, std, valid_mask = \
                get_median_depth(1 / render_pkg["depth"], render_pkg["opacity"], mask=valid_rgb, return_std=True)

            patches[:,:,2] = vo_initial_depth
            self.vo_patches_[self.vo_n] = patches
            self.vo_imap_[self.vo_n % self.vo_mem] = imap.squeeze()
            self.vo_gmap_[self.vo_n % self.vo_mem] = gmap.squeeze()
            self.vo_fmap1_[:, self.vo_n % self.vo_mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.vo_fmap2_[:, self.vo_n % self.vo_mem] = F.avg_pool2d(fmap[0], 4, 4)
                
            self.vo_n += 1
            self.vo_m += self.vo_M

            self.append_factors(*self.__edges_forw())
            self.append_factors(*self.__edges_back())

            for itr in range(1):
                self.vo_update()

        for itr in range(self.vo_n-self.vo_keyframe_index, self.vo_n):
            index = self.tstamps_[itr]
            pose = SE3(self.vo_poses_[itr]).matrix().detach().clone()
            self.cameras[index].update_RT(pose[0:3, 0:3].clone(), pose[0:3, 3].clone())

        index = [self.tstamps_[itr] for itr in [self.vo_n-4, self.vo_n-1, self.vo_n-2, self.vo_n-3, self.vo_n-5, self.vo_n-6, self.vo_n-7, self.vo_n-8]]
        prior_dense_depth = self.prior_depth_network.get_depth([self.cameras[itr] for itr in index]).cpu().numpy()[0]

        sparse_depth = self.get_sparse_depth(index[0])
        opt_dense_depth = optimize_depth(sparse_depth, prior_dense_depth)

        self.cameras[index[0]].depth = opt_dense_depth

        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=self.cameras[index[0]],
                gtcolor=self.cameras[index[0]].original_image,
                gtdepth=self.cameras[index[0]].depth
            )
        )

        return index[0]

    def is_map_keyframe(
        self,
        key_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[key_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, key_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [key_frame_idx] + window
        # remove frames which has little overlap with the key frame
        key_frame = self.cameras[key_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(key_frame.R, key_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, key_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", key_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, key_frame_idx, viewpoint):
        msg = ["map", key_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, key_frame_idx, viewpoint, depth_map):
        msg = ["init", key_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results and not self.eval_rendering:
                        eval_ate(
                            self.cameras,
                            self.map_kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                keyframe_idx = self.vo_tracking(cur_frame_idx)
                create_kf_vo = self.vo_keyframe()
                if not create_kf_vo:
                    self.cleanup(keyframe_idx)
                    cur_frame_idx += 1
                    continue

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=self.cameras[keyframe_idx],
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    cur_frame_idx += 1
                    continue

                render_pkg = render(
                    self.cameras[keyframe_idx], self.gaussians, self.pipeline_params, self.background)
                self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                last_keyframe_idx = self.current_window[0]
                check_time = (keyframe_idx - last_keyframe_idx) >= self.kf_interval
                create_kf_map = self.is_map_keyframe(
                    keyframe_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf_map = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )

                if create_kf_map:
                    self.current_window, removed = self.add_to_window(
                        keyframe_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    depth_map = self.add_map_keyframe(
                        keyframe_idx,
                        # depth=render_pkg["depth"],
                        # opacity=render_pkg["opacity"]
                    )
                    self.request_keyframe(
                        keyframe_idx, self.cameras[keyframe_idx], self.current_window, depth_map
                    )

                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and not self.eval_rendering
                    and create_kf_map
                    and len(self.map_kf_indices) > 10
                    # and len(self.map_kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.map_kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf_map:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
