#Goal is to run an array of different datasets, an array of different CNNs, with an array of different data augmentation methods. (maybe array of different loss)
#done by creating a bunch of cmdes

import sys
sys.path.append('/scratch/seoj4/installations')
import os

def main():
    #run on GLP
    glp_runstring="python ./GLPDepth/code/train.py --dataset nyudepthv2 --data_path ./GLPDepth/datasets/ --batch_size 12 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu cpu --save_result --save_model"
    os.system(glp_runstring)

if __name__ == '__main__':
    main()
