# A Tutorial Series on Deep Reinforcement Learning
## Author: Robert Tjarko Lange (TU Berlin)

This repository contains a series of tutorials on Deep Reinforcement Learning (DRL). This includes slides as well as experiments. Going forward I plan on adding exercises as well as complementary blog posts. So stay tuned!

### Deep Q-Learning (July 2019)

<img src="EXPERIMENTS_DQL/figures/dqn-timeline.jpeg" width="100%" height="100%" />

* [Slides](Deep_Q_Learning.pdf): Includes DQN, Double DQN, Prioritized Experience Replay & Dueling DQNs
* [Experiments](EXPERIMENTS_DQL): Provides code to implement all of the above.
* [Blog Post I/II](https://roberttlange.github.io/posts/2019/08/blog-post-5/): Covering all algorithms from Fitted Q-Learning to Categorical DQNs.
* Replicating the experiments:
    1. Create & activate a virtual env. Install the requirements.
    2. Afterwards you can run all experiments by executing:

        ```
        time bash run_experiments_dqn.sh dqn <cuda_device_id>
        time bash run_experiments_dqn.sh double-dqn <cuda_device_id>
        time bash run_experiments_dqn.sh per-dqn <cuda_device_id>
        time bash run_experiments_dqn.sh dueling-dqn <cuda_device_id>
        ```
    3. The visualizations for the different experiments as well as the mini double DQN illustration can be replicated by executing the notebook:

        ```
        jupyter notebook viz_results.ipynb
        ```

    4. Finally, in order to visualize an episode rollout of a DQN agent at different stages do the following:

        ```
        python train_dqn.py --SAVE_AGENT
        python enjoy_dense.py --AGENT 5000_MLP-DQN --TITLE 5000
        python enjoy_dense.py --AGENT 40000_MLP-DQN --TITLE 40000
        python enjoy_dense.py --AGENT 500000_MLP-DQN --TITLE 500000
        ```

        <img src="EXPERIMENTS_DQL/movies/DQN-Gridworld.gif" width="100%" height="100%" />


### Deep Policy Gradients (to be continued)
