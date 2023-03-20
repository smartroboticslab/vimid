#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

VI_CONFIG='../../../config/config_vimid.yaml'
DATA_DIR='/data/vimid/VD5'

cd $ROOT/build
./kfusion-main-openmp -a 0.6574,0.6126,-0.2949,-0.3248 -b -c 2 -f 0 -k 383.432,383.089,314.521,254.927 -l 1e-05 -m 0.075 -o $DATA_DIR/results/obj_poses/obj -p 0.5,0.5,0.4 -r 1 -s 10 -t 1 -v 512 -y 10,10,10 -z 1 -A -B 8 -F -G 0.01 -S $DATA_DIR/mask_RCNN -W $VI_CONFIG -X $DATA_DIR -I 0 -V 0.9 -u
