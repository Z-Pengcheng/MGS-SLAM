import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import BasicBlock, ConvBnReLU
from ..utils.generic_utils import upsample


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers


class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                num_output_channels=1,  
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([num_ch_enc[-2]/4, 
                                    num_ch_enc[-2]/4, 
                                    num_ch_enc[-2]/2, 
                                    num_ch_enc[-2]], dtype=np.int32)

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )
                if i != max_i:
                    self.convs[f"output_{i}"] = nn.Sequential(
                    BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                    nn.Conv2d(num_ch_out, self.num_output_channels, 1),
                    )

    def forward(self, level, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, (5-level)):
            max_i = 4 - j
            for i in range(max_i, (level-1), -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i-level])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i+1-level]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]

                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]

                if i == level:
                    depth_outputs[f"log_depth_pred_l{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]
            outputs = []

        return depth_outputs
    

class CostRegularization(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                depth_nums=[8, 8, 32],  
            ):
        super().__init__()

        self.num_output_channels = np.array([depth_nums[0], 
                                             depth_nums[1], 
                                             depth_nums[2]], dtype=np.int32)
        self.upsample_mode = 'nearest'
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([num_ch_enc[-2]/4, 
                                    num_ch_enc[-2]/4, 
                                    num_ch_enc[-2]/2, 
                                    num_ch_enc[-2]], dtype=np.int32)

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )
                if i != max_i:
                    self.convs[f"output_{i}"] = nn.Sequential(
                    BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                    nn.Conv2d(num_ch_out, self.num_output_channels[i], 1),
                    )

    def forward(self, level, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, (5-level)):
            max_i = 4 - j
            for i in range(max_i, (level-1), -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i-level])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i+1-level]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]

                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]

                if i == level:
                    depth_outputs[f"depth_pred_l{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]
            outputs = []

        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_outs[i]
            num_ch_out = num_ch_outs[i]

            self.convs[f"in_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1)
            
            self.convs[f"cat_conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )

            self.num_ch_enc.append(num_ch_out)

            if i is (self.num_blocks-1): continue
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_out, num_ch_outs[i+1],
                                                    stride=2)

    def forward(self, 
                level, 
                x, 
                img_feats):
        outputs = []
        for i in range(level, self.num_blocks):
            x = self.convs[f"in_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"cat_conv_{i}"](x)
            outputs.append(x)

            if i is (self.num_blocks-1): continue
            x = self.convs[f"ds_conv_{i}"](x)
            
        return outputs


class MatchingEncoder(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self):
        super(MatchingEncoder, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1),
                        ConvBnReLU(8, 8, 3, 2, 1),
                        ConvBnReLU(8, 8, 3, 1, 1))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2),
                        ConvBnReLU(16, 16, 3, 1, 1),
                        ConvBnReLU(16, 16, 3, 1, 1))

        self.conv2 = nn.Sequential( 
                        ConvBnReLU(16, 32, 5, 2, 2),
                        ConvBnReLU(32, 32, 3, 1, 1),
                        ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, 
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x) # (B, 8, H//2, W//2)
        conv1 = self.conv1(conv0) # (B, 16, H//4, W//4)
        conv2 = self.conv2(conv1) # (B, 32, H//8, W//8)
        feat2 = self.toplayer(conv2) # (B, 32, H//8, W//8)
        feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//4, W//4)
        feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H//2, W//2)

        # reduce output channels
        feat1 = self.smooth1(feat1) # (B, 16, H//4, W//4)
        feat0 = self.smooth0(feat0) # (B, 8, H//2, W//2)

        feats = {"level_0": feat0,
                 "level_1": feat1,
                 "level_2": feat2}

        return feats
