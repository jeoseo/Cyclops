#Goal is to run an array of different datasets, an array of different CNNs, with an array of different data augmentation methods. (maybe array of different loss)
#done by creating a bunch of cmdes

import sys
sys.path.append('/scratch/seoj4/installations')
import os

def main():
    #run on GLP with 1000 training examples
    glp_runstring="python ./GLPDepth/code/train.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu --save_result --save_model"
    os.system(glp_runstring+" --log_dir ./logs/GLP_log_1000_0 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/1000")



if __name__ == '__main__':
    main()
