from importlib import import_module

import torch
import torch.nn.functional as F
from torchvision import transforms

# The on-disk package name contains a hyphen, so it must be imported dynamically.
FastMVNet = import_module("fast-mvsnet.model").FastMVNet
imagenet_normalize = import_module("fast-mvsnet.utils.generic_utils").imagenet_normalize

@torch.no_grad()
class mvs_network:
    def __init__(self, config):
        self.config = config
        self.depth_network = FastMVNet.load_from_checkpoint(self.config["mvs_network"]["mvs_weight_path"])
        self.depth_network.cuda()
        self.depth_network.eval()

        self.max_depth = self.config["mvs_network"]["max_depth"]
        self.depth_image_height = 384
        self.depth_image_width = 512

    def get_depth(self, cameras):
        # intrinsics
        K = torch.eye(4, dtype=torch.float32).cuda()
        K[0, 0] = cameras[0].fx
        K[0, 2] = cameras[0].cx
        K[1, 1] = cameras[0].fy
        K[1, 2] = cameras[0].cy

        K[0] *= self.depth_image_width / float(cameras[0].image_width) / 2
        K[1] *= self.depth_image_height / float(cameras[0].image_height) / 2

        resize = transforms.Resize([self.depth_image_height, self.depth_image_width])

        # cur data
        cur_data = {}
        cur_img = resize(cameras[0].original_image)
        cur_img = imagenet_normalize(cur_img)
        cur_data["image_b3hw"] = cur_img.unsqueeze(0)
        Tw2c = torch.eye(4).cuda()
        Tw2c[:3, :3] = cameras[0].R
        Tw2c[:3, 3] = cameras[0].T
        cur_data["cam_T_world_b44"] = Tw2c.unsqueeze(0)
        cur_data["world_T_cam_b44"] = torch.linalg.inv(Tw2c).unsqueeze(0)
        for i in range(3):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = torch.linalg.inv(K_scaled)
            cur_data[f"K_l{i}_b44"] = K_scaled.unsqueeze(0)
            cur_data[f"invK_l{i}_b44"] = invK_scaled.unsqueeze(0)

        # src data
        src_data = {}
        src_img_list = []
        src_wtc_list = []
        src_ctw_list = []
        for i in range(1, len(cameras)):
            src_img = resize(cameras[i].original_image)
            src_img_list.append(imagenet_normalize(src_img))
            Tw2c = torch.eye(4).cuda()
            Tw2c[:3, :3] = cameras[i].R
            Tw2c[:3, 3] = cameras[i].T
            src_ctw_list.append(Tw2c)
            src_wtc_list.append(torch.linalg.inv(Tw2c))  
        src_data["image_b3hw"] = torch.stack(src_img_list).unsqueeze(0)
        src_data["cam_T_world_b44"] = torch.stack(src_ctw_list).unsqueeze(0)
        src_data["world_T_cam_b44"] = torch.stack(src_wtc_list).unsqueeze(0)
        for i in range(3):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = torch.linalg.inv(K_scaled)
            src_data[f"K_l{i}_b44"] = K_scaled.expand(1, len(cameras)-1, 4, 4)
            src_data[f"invK_l{i}_b44"] = invK_scaled.expand(1, len(cameras)-1, 4, 4)

        outputs = self.depth_network("test", cur_data, src_data)
        depth_mask = outputs["depth_pred_l0_b1hw"] < self.max_depth
        depth = outputs["depth_pred_l0_b1hw"] * depth_mask

        depth = F.interpolate(
                    depth, 
                    size=(cameras[0].image_height, cameras[0].image_width),
                    mode="nearest")
        
        return depth.detach().clone().squeeze(0)
