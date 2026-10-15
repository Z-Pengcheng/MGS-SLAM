import argparse
import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class Options():
    ################################ mgs slam ################################
    imagedir: str = "data/TUM/rgbd_dataset_freiburg3_long_office_household/rgb"
    calib: str = "data/TUM/rgbd_dataset_freiburg3_long_office_household/tum3.txt"

    ################################### VO ###################################
    # VO image size
    vo_image_width: int = 640
    vo_image_height: int = 480

    # VO config (increase for better accuracy)
    patches_per_frame: int = 96
    removal_window: int = 22
    optimization_window: int = 10 # BA
    patch_lifetime: int = 13
    buffer_size: int = 2048

    # threshold for keyframe removal
    keyframe_thresh: float = 15.0
    keyframe_index: int = 4

    # camera motion model
    motion_model: str = "DAMPED_LINEAR"
    motion_damping: float = 0.5

    # maybe use mixed precision for inference
    mixed_precision: bool = True

    gradient_bias: bool = False

    # vo weigth path
    vo_weigth_path: str = "weights/dpvo.pth"

    ############################## depth model ###############################
    # depth image size
    depth_image_width: int = 512
    depth_image_height: int = 384

    # fusion_max_depth
    fusion_max_depth: float = 3.0

    # depth weigth path
    depth_weigth_path: str = "weights/fast-mvsnet.ckpt"

    ################################ mapping #################################
    # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    scene_radius_depth_ratio: int = 3

    # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    mean_sq_dist_method: str = "projective"

    # mapping window size
    mapping_window_size: int = 8

    # mapping iterations
    mapping_iters: int = 1

    # mapping learn rate
    map_lrs_means3D: float = 0.0001
    map_lrs_rgb_colors: float = 0.0025
    map_lrs_unnorm_rotations: float = 0.001
    map_lrs_logit_opacities: float = 0.05
    map_lrs_log_scales: float = 0.001

    # use depth l1 loss?
    map_use_l1: bool = True

    # loss weights
    map_loss_im: float = 0.5
    map_loss_depth: float = 1.0

    # prune gaussians
    map_prune_gaussians: bool = True
    start_after: int = 0
    remove_big_after: int = 0
    stop_after: int = 20
    prune_every: int = 20
    removal_opacity_threshold: float = 0.005
    final_removal_opacity_threshold: float = 0.005
    reset_opacities: bool = False
    reset_opacities_every: int = 500 # Doesn't consider iter 0

    # for Addition of new Gaussians
    sil_thres: float = 0.5


class OptionsHandler():
    def __init__(self):
        self.options = Options()

        self.parser = argparse.ArgumentParser(description="FastMVNet Options")
        self.parser.add_argument('--config_file', type=str, default=None)

        self.populate_argparse()

    def populate_argparse(self):
        for field_name in self.options.__dataclass_fields__.keys():
            field_info = self.options.__dataclass_fields__[field_name]
            if field_info.type == bool:
                self.parser.add_argument(f'--{field_name}', type=bool)
            else:
                self.parser.add_argument(
                                        f'--{field_name}', 
                                        type=field_info.type, 
                                        default = None,
                                    )

    def parse_and_merge_options(self, config_filepaths=None, ignore_cl_args=False):
        if not ignore_cl_args:
            cl_args = self.parser.parse_args()

        if config_filepaths is not None:
            if isinstance(config_filepaths, list):
                for config_filepath in config_filepaths:
                    config_options = OptionsHandler.load_options_from_yaml(
                                                                config_filepath)
                    self.merge_config_options(config_options)
            else:
                config_options = OptionsHandler.load_options_from_yaml(
                                                               config_filepaths)
                self.merge_config_options(config_options)

        elif (not ignore_cl_args and cl_args.config_file is not None):
            if cl_args.config_file is not None:
                config_options = OptionsHandler.load_options_from_yaml(
                                                            cl_args.config_file)
                self.merge_config_options(config_options)
            
        else:
            print("Not reading from a config_file.")

        if not ignore_cl_args:
            self.merge_cl_args(cl_args)


    def merge_config_options(self, config_options):
        for field_name in config_options.__dict__.keys():
            value = config_options.__getattribute__(field_name)
            self.options.__setattr__(field_name, value)

    def merge_cl_args(self, cl_args):
        for arg_pair in cl_args._get_kwargs():
            if arg_pair[0] == "config_file":
                continue
            if arg_pair[1] is not None:
                self.options.__setattr__(arg_pair[0], arg_pair[1])

    @staticmethod
    def load_options_from_yaml(config_filepath):
        stream = open(config_filepath, 'r')
        return yaml.load(stream, Loader=yaml.Loader)
    
    @staticmethod
    def save_options_as_yaml(config_filepath, options):
        with open(config_filepath, 'w') as outfile:
            yaml.dump(options, outfile, default_flow_style=False)
                
    def pretty_print_options(self):
        print("########################### Options ###########################")
        print("")
        for field_name in self.options.__dataclass_fields__.keys():
            print("    ", field_name + ":", self.options.__getattribute__(field_name))
        print("")
        print("###############################################################")
