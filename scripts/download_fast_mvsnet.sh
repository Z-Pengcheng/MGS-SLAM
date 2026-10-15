#!/usr/bin/env bash
set -e

mkdir -p weights
gdown --continue 'https://drive.google.com/file/d/1zFFsi1dDXRZC0U-i_nQDiWYdMlI8fUbR/view?usp=sharing' -O weights/fast-mvsnet.ckpt
