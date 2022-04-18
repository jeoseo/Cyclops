#Goal is to run an array of different datasets, an array of different CNNs, with an array of different data augmentation methods. (maybe array of different loss)
#done by creating a bunch of cmdes
import time
import subprocess
import sys
import os

def main():
    #time on 100_0 examples, extrapolate rate out to 6 hours
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py 100 1")
    glp_runstring="python ./GLPDepth/code/train.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --workers 1 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu"
    with os.popen(glp_runstring+" --epochs 2 --log_dir ./logs/GLP_100_1 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/100_1") as f:
        last_line = f.readlines()[-1]
    runtime=float(last_line)
    print(runtime)

    #run for 6 hours
    hours=3
    train_subset_size=int(100*hours*3600*2/25/runtime)
    train_str=str(train_subset_size)+"_100"
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py "+str(train_subset_size)+" 100")
    os.system(glp_runstring+" --save_model --log_dir ./logs/GLP_log_"+train_str+" --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/"+train_str)


    #inference and evaluate
    glp_teststring="python ./GLPDepth/code/test.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --save_visualize --do_evaluate --max_depth 10.0 --max_depth_eval 10.0 --gpu_or_cpu gpu"
    os.system(glp_teststring+" --result_dir ./logs/GLP_"+train_str+" --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/"+train_str+" --ckpt_dir ./logs/GLP_"+train_str+"/epoch_25_model.ckpt")

    #lapdepth 
   #lap_runstring="python ./LapDepth/train.py --trainfile_nyu ./GLPDepth/code/dataset/filenames/nyudepthv2/10000/train_subset.txt --testfile_nyu ./GLPDepth/code/dataset/filenames/nyudepthv2/1000/test_subset.txt --batch_size 12 --workers 1 --dataset NYU --data_path ./datasets/nyu_depth_v2 --epochs 25"

if __name__ == '__main__':
    main()
