###############################################################################
# DQN EXPERIMENTS
###############################################################################
# DQN - ER Capacity Experiments
python train_dqn.py -s -n_runs 8 -capacity 100 -fname dqn_capacity_100
python train_dqn.py -s -n_runs 8 -capacity 1000 -fname dqn_capacity_1000
python train_dqn.py -s -n_runs 8 -capacity 5000 -fname dqn_capacity_5000
# DQN - Batch Size Experiments
python train_dqn.py -s -n_runs 8 -train_batch 32 -fname dqn_batchsize_32
python train_dqn.py -s -n_runs 8 -train_batch 64 -fname dqn_batchsize_64
python train_dqn.py -s -n_runs 8 -train_batch 128 -fname dqn_batchsize_128
# DQN - Learning Rate Experiments
python train_dqn.py -s -n_runs 8 -l_r 0.01 -fname dqn_lrate_1e3
python train_dqn.py -s -n_runs 8 -l_r 0.001 -fname dqn_lrate_1e4
python train_dqn.py -s -n_runs 8 -l_r 0.0001 -fname dqn_lrate_1e5
# DQN - Hard vs Soft Updates
python train_dqn.py -s -n_runs 8 -update_upd 1 -fname dqn_hard_update_1
python train_dqn.py -s -n_runs 8 -update_upd 100 -fname dqn_hard_update_100
python train_dqn.py -s -n_runs 8 -soft_tau 0.01 -fname dqn_soft_update_001
python train_dqn.py -s -n_runs 8 -soft_tau 0.05 -fname dqn_soft_update_005
###############################################################################
# DOUBLE DQN EXPERIMENTS
###############################################################################
python train_dqn.py -s -n_runs 8 --DOUBLE -soft_tau 0.01 -fname ddqn_soft_update_001
python train_dqn.py -s -n_runs 8 --DOUBLE -soft_tau 0.05 -fname ddqn_soft_update_005
python train_dqn.py -s -n_runs 8 --DOUBLE -soft_tau 0.1 -fname ddqn_soft_update_01
python train_dqn.py -s -n_runs 8 --DOUBLE -gamma 0.9 -fname ddqn_gamma_09
python train_dqn.py -s -n_runs 8 --DOUBLE -gamma 0.95 -fname ddqn_gamma_095
python train_dqn.py -s -n_runs 8 --DOUBLE -gamma 0.95 -fname ddqn_gamma_099
###############################################################################
# PRIORITIZED ER DQN EXPERIMENTS
###############################################################################
###############################################################################
# DUELING DQN EXPERIMENTS
###############################################################################
python train_dqn.py -s -n_runs 8 -agent MLP-DUELING -fname dueling_dqn
python train_dqn.py -s -n_runs 8 -agent MLP-DUELING --DOUBLE -fname double_dueling_dqn
python train_dqn.py -s -n_runs 8 -agent MLP-DUELING --PER -fname per_dueling_dqn
python train_dqn.py -s -n_runs 8 -agent MLP-DUELING --DOUBLE --PER -fname per_double_dueling_dqn
