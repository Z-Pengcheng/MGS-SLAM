# MGS-SLAM: Monocular Sparse Tracking and Gaussian Mapping with Depth Smooth Regularization

<h3 align="center"><strong>IEEE Robotics and Automation Letters (RA-L), 2025</strong></h3>

<h3 align="center"><a href="https://ieeexplore.ieee.org/document/10679906/">Paper</a> | <a href="https://arxiv.org/abs/2405.06241">ArXiv</a> | <a href="https://www.bilibili.com/video/BV1zM4m1C761">Bilibili</a> | <a href="https://youtu.be/twwELF3LgZs">YouTube</a></h3>

<p align="center"><video src="media/MGS-SLAM.mp4" controls width="800"></video></p>

MGS-SLAM is a monocular dense SLAM system that jointly optimizes sparse visual odometry and a 3D Gaussian scene representation. A fast multi-view stereo network estimates depth over visual odometry keyframe windows to guide mapping. Depth smooth regularization and Sparse-Dense Adjustment Ring (SDAR) improve geometric consistency and maintain a consistent scale between tracking and mapping.

## Installation

Clone the repository and initialize its submodules:

```bash
git clone https://github.com/Z-Pengcheng/MGS-SLAM.git
cd MGS-SLAM
git submodule update --init --recursive
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate MGS-SLAM
```

Install [DPVO](https://github.com/princeton-vl/DPVO) at commit `58338357611e7cf3563ed90f82168da7f8741d24` into the same environment:

```bash
cd ..
git clone https://github.com/princeton-vl/DPVO.git
cd DPVO
git checkout 58338357611e7cf3563ed90f82168da7f8741d24
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
python -m pip install .
cd ../MGS-SLAM
```

## Running

Download the datasets from the repository root:

```bash
bash scripts/download_replica.sh
bash scripts/download_tum.sh
bash scripts/download_icl_nuim.sh
```

Download the Fast-MVSNet checkpoint to `weights/fast-mvsnet.ckpt`:

```bash
bash scripts/download_fast_mvsnet.sh
```

Run MGS-SLAM and evaluate it on a sequence:

**Replica**

```bash
python slam.py --config configs/mono/replica/room0.yaml
```

**TUM RGB-D**

```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

**ICL-NUIM**

```bash
python slam.py --config configs/mono/icl_nuim/living_room_tk0.yaml
```

## Acknowledgement

This work incorporates many open-source codes. We extend our gratitude to the authors of the software.

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [SimpleRecon](https://github.com/nianticlabs/simplerecon)

## Citation

```bibtex
@ARTICLE{10679906,
  author={Zhu, Pengcheng and Zhuang, Yaoming and Chen, Baoquan and Li, Li and Wu, Chengdong and Liu, Zhanlin},
  journal={IEEE Robotics and Automation Letters},
  title={MGS-SLAM: Monocular Sparse Tracking and Gaussian Mapping With Depth Smooth Regularization},
  year={2024},
  volume={9},
  number={11},
  pages={9486-9493},
  keywords={Simultaneous localization and mapping;Visual odometry;Accuracy;Three-dimensional displays;Image reconstruction;Rendering (computer graphics);Cameras;SLAM;mapping;3D Gaussian splatting},
  doi={10.1109/LRA.2024.3460431}
}
```
