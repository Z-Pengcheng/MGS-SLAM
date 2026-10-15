import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.geometry_utils import BackprojectDepth, Project3D


class DotCostVolumeManager(nn.Module):
    """
    Class to build a cost volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    collapsing over views by taking a dot product between each source and 
    reference feature, before summing over source views at each pixel location. 
    The final tensor is size batch_size x num_depths x H x  W tensor.
    """

    def __init__(self):

        super().__init__()
        
        self.backprojector = BackprojectDepth()
        self.projector = Project3D()        

    def warp_features(
                    self, 
                    src_feats, 
                    src_extrinsics, 
                    src_Ks, 
                    cur_invK, 
                    depth_plane_b1hw, 
                    batch_size, 
                    num_src_frames, 
                    num_feat_channels,
                    uv_scale,
                ):
        """
        Warps every soruce view feature to the current view at the depth 
        plane defined by depth_plane_b1hw.
        """
        
        # backproject points at that depth plane to the world, where the 
        # world is really the current view.
        _, _, h, w = depth_plane_b1hw.shape
        world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
        world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, 
                                                                        dim=0)
        
        # project these points down to each source frame
        cam_points_B3N = self.projector(
                                    world_points_B4N, 
                                    src_Ks.view(-1, 4, 4), 
                                    src_extrinsics.view(-1, 4, 4)
                                )

        cam_points_B3hw = cam_points_B3N.view(-1, 3, h, w)
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        depths = cam_points_B3hw[:, 2:]

        uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1

        src_feat_warped = F.grid_sample(
                                    input=src_feats.view(
                                        -1, 
                                        num_feat_channels, 
                                        h, 
                                        w
                                    ),
                                    grid=uv_coords.type_as(src_feats),
                                    padding_mode='zeros',
                                    mode='bilinear',
                                    align_corners=False,
                                )

        # Reshape tensors to "unbatch"
        src_feat_warped = src_feat_warped.view(
                                            batch_size,
                                            num_src_frames,
                                            num_feat_channels,
                                            h,
                                            w,
                                        )

        depths = depths.view(
                        batch_size,
                        num_src_frames,
                        h,
                        w
                    )

 
        mask_b = depths > 0
        mask = mask_b.type_as(src_feat_warped)
                             
        return world_points_B4N, depths, src_feat_warped, mask


    def build_cost_volume(
                        self, 
                        cur_feats: Tensor,
                        src_feats: Tensor,
                        src_extrinsics: Tensor,
                        src_Ks: Tensor,
                        cur_invK: Tensor,
                        depth_planes_bdhw: Tensor,
                    ):
        """
        build cost volume
        """

        batch_size, num_src_frames, num_feat_channels, h, w = src_feats.shape
        b, num_depth_bins, _, _ = depth_planes_bdhw.shape

        uv_scale = torch.tensor(
                                [1 / w, 
                                1 / h], 
                                dtype=src_extrinsics.dtype, 
                                device=src_extrinsics.device
                            ).view(1, 1, 1, 2)

        # Intialize the cost volume and the counts
        all_dps = []

        # loop through depth planes
        for depth_id in range(num_depth_bins):

            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            _, _, src_feat_warped, mask = self.warp_features(
                                                        src_feats, 
                                                        src_extrinsics, 
                                                        src_Ks, 
                                                        cur_invK, 
                                                        depth_plane_b1hw, 
                                                        batch_size, 
                                                        num_src_frames, 
                                                        num_feat_channels,
                                                        uv_scale,
                                                    )


            # Compute the dot product between cur and src features
            dot_product_bkhw = torch.sum(
                                        src_feat_warped * 
                                            cur_feats.unsqueeze(1), 
                                        dim=2,
                                ) * mask

            # Sum over the frames
            dot_product_b1hw = dot_product_bkhw.sum(dim=1, keepdim=True)
            all_dps.append(dot_product_b1hw)

        cost_volume = torch.cat(all_dps, dim=1)

        return cost_volume


    def forward(
            self, 
            cur_feats, 
            src_feats, 
            src_extrinsics, 
            src_Ks, 
            cur_invK,
            depth_planes_bdhw,
        ):
        """ Runs the cost volume and gets the lowest cost result """
        cost_volume = self.build_cost_volume(
                                            cur_feats=cur_feats,
                                            src_feats=src_feats,
                                            src_extrinsics=src_extrinsics,
                                            src_Ks=src_Ks,
                                            cur_invK=cur_invK,
                                            depth_planes_bdhw=depth_planes_bdhw,
                                        )

        return cost_volume
