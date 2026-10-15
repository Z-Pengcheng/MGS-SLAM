import cv2
import numpy as np

import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, use_depth_l1, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, use_depth_l1)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()



def _tensor_size(t):
    return t.size()[0] * t.size()[1] * t.size()[2]
 
 
def tv_loss(x):
    h_x = x.size()[1]
    w_x = x.size()[2]
    count_h = _tensor_size(x[:, 1:, :])
    count_w = _tensor_size(x[:, :, 1:])
    h_tv = torch.pow((x[:, 1:, :] - x[:, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, 1:] - x[:, :, :w_x - 1]), 2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)


# def image2canny(image, thres1, thres2, isEdge1=True):
#     """ image: (H, W, 3)"""
#     canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
#     if not isEdge1:
#         canny_mask = 1. - canny_mask
#     return canny_mask.float()


# with torch.no_grad():
#     kernelsize=3
#     conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize//2))
#     kernel = torch.tensor([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]).reshape(1,1,kernelsize,kernelsize)
#     conv.weight.data = kernel #torch.ones((1,1,kernelsize,kernelsize))
#     conv.bias.data = torch.tensor([0.])
#     conv.requires_grad_(False)
#     conv = conv.cuda()


# def nearMean_map(array, mask, kernelsize=3):
#     """ array: (H,W) / mask: (H,W) """
#     cnt_map = torch.ones_like(array)

#     nearMean_map = conv((array * mask)[None,None])
#     cnt_map = conv((cnt_map * mask)[None,None])
#     nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
#     return nearMean_map


def get_loss_mapping_rgbd(config, image, depth, viewpoint, use_depth_l1, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    ltv = tv_loss(depth)

    if use_depth_l1:
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=image.device
        )[None]
        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

        return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean() + 1.0*ltv

    # depth = depth * depth_pixel_mask
    # gt_depth = gt_depth * depth_pixel_mask
    # tmp = (gt_depth - depth).sum()/depth_pixel_mask.sum()
    # diff = torch.abs((depth - gt_depth + tmp))
    # l1_depth = diff

    # depth = depth.squeeze(0)
    # canny_mask = image2canny(gt_image.permute(1,2,0), 50, 150, isEdge1=False).detach().cuda()
    # depth_mask = (depth>0).detach()
    # nearDepthMean_map = nearMean_map(depth, canny_mask*depth_mask, kernelsize=3)
    # l2_depth_smooth = ((nearDepthMean_map - depth*depth_mask) ** 2).mean() * 1.0

    return alpha * l1_rgb.mean() + 1.0*ltv


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def optimize_depth(sparse_depth, prior_dense_depth):
    valid_mask = np.nonzero(sparse_depth)
    std1 = sparse_depth[valid_mask].std()
    std3 = prior_dense_depth[valid_mask].std()
    mean1 = sparse_depth[valid_mask].mean()
    mean2 = prior_dense_depth.mean()
    mean3 = prior_dense_depth[valid_mask].mean()

    alpha = std1 / std3
    beta = mean2*(mean1/mean3 - alpha)

    opt_dense_depth = alpha * prior_dense_depth + beta

    return opt_dense_depth
