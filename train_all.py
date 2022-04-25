#Goal is to run an array of different datasets, an array of different CNNs, with an array of different data augmentation methods. (maybe array of different loss)
#done by creating a bunch of cmds
import time
import subprocess
import sys
import os

def GLP():
    
    ##GLP
    
    #time on 100_1 examples, extrapolate rate out to 12 hours
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py 100 1")
    glp_runstring="python ./GLPDepth/code/train.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --workers 1 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu"
    with os.popen(glp_runstring+" --epochs 2 --log_dir ./logs/GLP_100_1 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/100_1") as f:
        last_line = f.readlines()[-1]
    runtime=float(last_line)
    print(runtime)

    #run for 12 hours
    hours=12
    train_subset_size=int(100*hours*3600*2/25/runtime)
    train_str=str(train_subset_size)+"_100"
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py "+str(train_subset_size)+" 100")
    os.system(glp_runstring+" --save_last_model --log_dir ./logs/GLP_"+train_str+" --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/"+train_str)

    #inference and evaluate
    glp_teststring="python ./GLPDepth/code/test.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --save_visualize --save_eval_pngs --do_evaluate --max_depth 10.0 --max_depth_eval 10.0 --gpu_or_cpu gpu"
    os.system(glp_teststring+" --result_dir ./logs/GLP_"+train_str+"/results --filenames_path ./GLPDepth/code/dataset/filenames/SUN_RGBD --ckpt_dir ./logs/GLP_"+train_str+"/epoch_25_model.ckpt")


def BTS():
    
    ##BTS

    #time on 100_1 examples, extrapolate rate out to 12 hours
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py 100 1")
    bts_runstring="python ./GLPDepth/code/train_bts.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --workers 1 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu"
    with os.popen(bts_runstring+" --epochs 2 --log_dir ./logs/bts_100_1 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/100_1") as f:
        last_line = f.readlines()[-1]
    runtime=float(last_line)
    print(runtime)

    #run for 12 hours, with 100 examples for validation
    hours=12
    train_subset_size=int(100*hours*3600*2/25/runtime)
    train_str=str(train_subset_size)+"_100"
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py "+str(train_subset_size)+" 100")
    os.system(bts_runstring+" --save_last_model --log_dir ./logs/bts_"+train_str+" --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/"+train_str)

    #inference and evaluate
    bts_teststring="python ./GLPDepth/code/test_bts.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --save_visualize  --save_eval_pngs --do_evaluate --max_depth 10.0 --max_depth_eval 10.0 --gpu_or_cpu gpu"
    os.system(bts_teststring+" --result_dir ./logs/bts_"+train_str+"/results --filenames_path ./GLPDepth/code/dataset/filenames/SUN_RGBD --ckpt_dir ./logs/bts_"+train_str+"/epoch_25_model.ckpt")    


def LAP():
    
    ##Lapdepth

    #time on 100_1 examples, extrapolate rate out to 12 hours
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py 100 1")
    lap_runstring="python ./GLPDepth/code/train_lap.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --batch_size 12 --workers 1 --max_depth 10.0 --max_depth_eval 10.0  --gpu_or_cpu gpu"
    with os.popen(lap_runstring+" --epochs 2 --log_dir ./logs/lap_100_1 --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/100_1") as f:
        last_line = f.readlines()[-1]
    runtime=float(last_line)
    print(runtime)

    #run for 12 hours, with 100 examples for validation
    hours=12
    train_subset_size=int(100*hours*3600*2/25/runtime)
    train_str=str(train_subset_size)+"_100"
    os.system("python ./GLPDepth/code/dataset/filenames/nyudepthv2/train_test_set_generator.py "+str(train_subset_size)+" 100")
    os.system(lap_runstring+" --save_last_model --log_dir ./logs/lap_"+train_str+" --filenames_path ./GLPDepth/code/dataset/filenames/nyudepthv2/"+train_str)

    #inference and evaluate
    lap_teststring="python ./GLPDepth/code/test_lap.py --dataset nyudepthv2 --data_path ./datasets/nyu_depth_v2 --save_visualize  --save_eval_pngs --do_evaluate --max_depth 10.0 --max_depth_eval 10.0 --gpu_or_cpu gpu"
    os.system(lap_teststring+" --result_dir ./logs/lap_"+train_str+"/results --filenames_path ./GLPDepth/code/dataset/filenames/SUN_RGBD --ckpt_dir ./logs/lap_"+train_str+"/epoch_25_model.ckpt")    



def main():
    
    GLP()
    BTS()
    LAP()

if __name__ == '__main__':
    main()
