###############################################################################
# DQN EXPERIMENTS
###############################################################################
# DQN - ER Capacity Experiments
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -capacity 5000 -fname dqn_capacity_5000
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -capacity 20000 -fname dqn_capacity_20000
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -capacity 40000 -fname dqn_capacity_40000
# DQN - Batch Size Experiments
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -train_batch 32 -fname dqn_batchsize_32
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -train_batch 64 -fname dqn_batchsize_64
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -train_batch 128 -fname dqn_batchsize_128
# DQN - Learning Rate Experiments
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -l_r 0.01 -fname dqn_lrate_1e3
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -l_r 0.001 -fname dqn_lrate_1e4
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -l_r 0.0001 -fname dqn_lrate_1e5
# DQN - Hard vs Soft Updates
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -update_upd 1 -fname dqn_hard_update_1
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -update_upd 500 -fname dqn_hard_update_100
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -soft_tau 0.01 -fname dqn_soft_update_001
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 -soft_tau 0.05 -fname dqn_soft_update_005
###############################################################################
# DOUBLE DQN EXPERIMENTS
###############################################################################
# DDQN - Updates of Target Networks
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -soft_tau 0.01 -fname ddqn_soft_update_001
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -soft_tau 0.05 -fname ddqn_soft_update_005
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -soft_tau 0.1 -fname ddqn_soft_update_01
# DDQN - Discounting Future Bias
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -gamma 0.9 -fname ddqn_gamma_09
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -gamma 0.95 -fname ddqn_gamma_095
CUDA_VISIBLE_DEVICES=0,1 python train_dqn.py -s -n_runs 6 --DOUBLE -gamma 0.99 -fname ddqn_gamma_099
