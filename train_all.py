#Goal is to run an array of different datasets, an array of different CNNs, with an array of different data augmentation methods. (maybe array of different loss)
#done by creating a bunch of cmdes
import time

import sys
sys.path.append('/scratch/seoj4/installations')
import os

def main():
    #run on GLP with 1000 training examples
    glp_runstring="python ./GLPDepth/code/train.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --workers 1 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu --save_result --save_model"
    os.system(glp_runstring+" --log_dir ./logs/GLP_log_1000_0 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/1000")



    #inference with each
    glp_teststring="python ./GLPDepth/code/test.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/1000 --ckpt_dir ./logs/GLP_10000_0/epoch_25_model.ckpt --save_visualize  --max_depth 10.0 --max_depth_eval 10.0 --gpu_or_cpu gpu"
    os.system(glp_teststring)

    #lapdepth 
    lap_runstring="python ./LapDepth/train.py --use_dense_depth False --batch_size 12 --workers 1 --dataset NYU --data_path ./datasets/nyu_depth_v2 --epochs 25"

if __name__ == '__main__':
    main()
