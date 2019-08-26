#!/bin/bash
today=`date '+%Y_%m_%d_%H_%M_%S'`
TB=FROSI_VISNET_check
set -ex
nohup python -u train_vis.py --lr 0.00001 --gpu_ids 1 \
--batch_size 1 --name maps_visnet_$TB \
--dataroot ./datasets/datasets/FROSI/Fog \
--TBoardX $TB --save_epoch_freq 1 \
--niter 1 --niter_decay 0 --model visnet --dataset_mode frosi &> ./outputmd/output_$TB.md & \

tail -f ./outputmd/output_$TB.md
