###############################################################################
# PRIORITIZED ER DQN EXPERIMENTS
###############################################################################
# PER - Priority Distribution Temperature
CUDA_VISIBLE_DEVICES=2,3  python train_dqn.py -s -n_runs 6 --PER --ALPHA 0.2 -fname per_dqn_alpha_02
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER --ALPHA 0.5 -fname per_dqn_alpha_05
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER --ALPHA 0.8 -fname per_dqn_alpha_08
# PER - Importance Sampling Start Beta
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER --BETA_START 0.2 -fname per_dqn_beta_02
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER --BETA_START 0.5 -fname per_dqn_beta_05
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER --BETA_START 0.8 -fname per_dqn_beta_08
# PER - Buffer Capacity
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER -capacity 5000 -fname per_dqn_capacity_5000
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER -capacity 20000 -fname per_dqn_capacity_20000
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 --PER -capacity 40000 -fname per_dqn_capacity_40000
###############################################################################
# DUELING DQN EXPERIMENTS
###############################################################################
# Dueling DQN - Network Capacity
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --HIDDEN_SIZE 64 -fname dueling_dqn_64
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --HIDDEN_SIZE 128 -fname dueling_dqn_128
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --HIDDEN_SIZE 256 -fname dueling_dqn_256
# Dueling DQN - Combining the improvements
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING -fname dueling_dqn
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --DOUBLE -fname double_dueling_dqn
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --PER -fname per_dueling_dqn
CUDA_VISIBLE_DEVICES=2,3 python train_dqn.py -s -n_runs 6 -agent MLP-DUELING --DOUBLE --PER -fname per_double_dueling_dqn
