mkdir -p log
now=$(date +"%Y%m%d_%H%M%S_7B20")
srun --partition=3DV --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=samp --kill-on-bad-exit=1 /mnt/lustre/huangzhaoyang/anaconda3/envs/SenseLocNet/bin/python /mnt/lustre/huangzhaoyang/Code/MapNet/mapnet/train.py 2>&1|tee log/train-$now.log &

