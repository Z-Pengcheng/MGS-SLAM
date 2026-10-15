#!/usr/bin/env bash
set -e

mkdir -p datasets/icl-nuim
cd datasets/icl-nuim

wget -c https://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz
tar -xvzf living_room_traj0_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz
tar -xvzf living_room_traj1_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz
tar -xvzf living_room_traj2_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz
tar -xvzf living_room_traj3_frei_png.tar.gz

wget -c https://www.doc.ic.ac.uk/~ahanda/traj0_frei_png.tar.gz
tar -xvzf traj0_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/traj1_frei_png.tar.gz
tar -xvzf traj1_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz
tar -xvzf traj2_frei_png.tar.gz
wget -c https://www.doc.ic.ac.uk/~ahanda/traj3_frei_png.tar.gz
tar -xvzf traj3_frei_png.tar.gz
