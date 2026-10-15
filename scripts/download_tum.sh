#!/usr/bin/env bash
set -e

mkdir -p datasets/tum
cd datasets/tum

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
tar -xvzf rgbd_dataset_freiburg1_desk2.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz
tar -xvzf rgbd_dataset_freiburg1_plant.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
tar -xvzf rgbd_dataset_freiburg2_xyz.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz
