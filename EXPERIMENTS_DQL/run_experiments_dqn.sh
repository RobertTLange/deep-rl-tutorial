if [[ "$*" == "dqn" ]]
then
    echo "Run DQN Experiments"
    ###############################################################################
    # DQN EXPERIMENTS
    ###############################################################################
    # DQN - ER Capacity Experiments
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -capacity 5000 -fname dqn_capacity_5000 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -capacity 20000 -fname dqn_capacity_20000 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -capacity 40000 -fname dqn_capacity_40000 -n_upd 1500000
    # DQN - Batch Size Experiments
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -train_batch 32 -fname dqn_batchsize_32 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -train_batch 64 -fname dqn_batchsize_64 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -train_batch 128 -fname dqn_batchsize_128 -n_upd 1500000
    # DQN - Learning Rate Experiments
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -l_r 0.01 -fname dqn_lrate_1e3 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -l_r 0.001 -fname dqn_lrate_1e4 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -l_r 0.0001 -fname dqn_lrate_1e5 -n_upd 1500000
    # DQN - Hard vs Soft Updates
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -update_upd 1 -fname dqn_hard_update_1 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -update_upd 500 -fname dqn_hard_update_100 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -soft_tau 0.01 -fname dqn_soft_update_001 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -soft_tau 0.05 -fname dqn_soft_update_005 -n_upd 1500000
elif [[ "$*" == "double-dqn" ]]
then
    echo "Run Double DQN Experiments"
    ###############################################################################
    # DOUBLE DQN EXPERIMENTS
    ###############################################################################
    # DDQN - Updates of Target Networks
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -soft_tau 0.01 -fname ddqn_soft_update_001 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -soft_tau 0.05 -fname ddqn_soft_update_005 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -soft_tau 0.1 -fname ddqn_soft_update_01 -n_upd 1500000
    # DDQN - Discounting Future Bias
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -gamma 0.9 -fname ddqn_gamma_09 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -gamma 0.95 -fname ddqn_gamma_095 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --DOUBLE -gamma 0.99 -fname ddqn_gamma_099 -n_upd 1500000
elif [[ "$*" == "per-dqn" ]]
then
    echo "Run Prioritized Experience Replay Experiments"
    ###############################################################################
    # PRIORITIZED ER DQN EXPERIMENTS
    ###############################################################################
    # PER - Priority Distribution Temperature
    CUDA_VISIBLE_DEVICES=$2  python train_dqn.py -s -n_runs 5 --PER --ALPHA 0.2 -fname per_dqn_alpha_02 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER --ALPHA 0.5 -fname per_dqn_alpha_05 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER --ALPHA 0.8 -fname per_dqn_alpha_08 -n_upd 1500000
    # PER - Importance Sampling Start Beta
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER --BETA_START 0.2 -fname per_dqn_beta_02 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER --BETA_START 0.5 -fname per_dqn_beta_05 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER --BETA_START 0.8 -fname per_dqn_beta_08 -n_upd 1500000
    # PER - Buffer Capacity
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER -capacity 5000 -fname per_dqn_capacity_5000 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER -capacity 20000 -fname per_dqn_capacity_20000 -n_upd 1500000
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 --PER -capacity 40000 -fname per_dqn_capacity_40000 -n_upd 1500000
elif [[ "$*" == "dueling-dqn" ]]
then
    echo "Run Dueling DQN Experiments"
    ###############################################################################
    # DUELING DQN EXPERIMENTS
    ###############################################################################
    # Dueling DQN - Network Capacity
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --HIDDEN_SIZE 64 -fname dueling_dqn_64
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --HIDDEN_SIZE 128 -fname dueling_dqn_128
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --HIDDEN_SIZE 256 -fname dueling_dqn_256
    # Dueling DQN - Combining the improvements
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING -fname dueling_dqn
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --DOUBLE -fname double_dueling_dqn
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --PER -fname per_dueling_dqn
    CUDA_VISIBLE_DEVICES=$2 python train_dqn.py -s -n_runs 5 -agent MLP-DUELING --DOUBLE --PER -fname per_double_dueling_dqn

fi
