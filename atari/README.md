# Atari Experiments #


This directory contains PyTorch code and learned reward functions for "Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations". ICML 2019. 

If you prefer TensorFlow over PyTorch, see https://github.com/msinto93/T-REX-IRL for a good implementation.


## T-REX reward learning for Atari ##

First download PPO checkpoints into the models directory
https://github.com/dsbrown1331/learning-rewards-of-learners/releases/tag/atari25

The main file to run is:
LearnAtariReward.py

Here's an example of how to run it. 

```python LearnAtariReward.py --env_name breakout --reward_model_path ./learned_models/breakout_test.params --models_dir .```

To learn reward for enduro run

```
python LearnAtariReward.py --env_name enduro --reward_model_path ./learned_models/enduro_test.params --models_dir . --num_trajs 2000 --num_snippets 0
```

## Atari Grand Challenge demos ##

First download Atari Grand Challenge demos
https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/full.tar.gz

Install Dataset API following the instructions here:
https://github.com/dsbrown1331/atarigrandchallenge

To learn a reward function use LearnAtariRewardAGC.py

For example, assuming you downloaded the Atari Grand Challenge data set into /home/data/atari_v1/ you would run:

```python LearnAtariRewardAGC.py --env_name spaceinvaders --data_dir /home/data/atari_v1/ --reward_model_path ./learned_models/test_agc_spaceinvaders.params```

## Human MTurk rankings ##

The processed files from the Turkers are available in human_labels/

To run reward learning with human rankings use LearnAtariMTurkRankings.py

For example

```python LearnAtariMTurkRankings.py --env_name breakout --reward_model_path ./learned_models/breakout_mturk_test.params --models_dir ~/Code/learning-rewards-of-learners/learner/```

For Enduro run

```python LearnAtariMTurkRankings.py --env_name enduro --reward_model_path ./learned_models/enduro_mturk_test.params --models_dir . --num_trajs 2000 --num_snippets 0```



## Visualizing learned reward functions ##
The visualization script is VisualizeAtariLearnedReward.py

To generate plots run:
```python VisualizeAtariLearnedReward.py --env_name breakout --models_dir . --reward_net_path ./learned_models/icml_learned_rewards/breakout_progress_masking.params --save_fig_dir ./viz```

## RL on learned reward function ##

Given a trained reward network you can run RL as follows:

First baselines must be installed

```
cd baselines
pip install -e .
```

Then you can run RL as follows:

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=[your_log_dir_here] python -m baselines.run --alg=ppo2 --env=[Atari env here] --custom_reward pytorch --custom_reward_path [path_to_learned_reward_model] --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9
```


For example

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/tflogs python -m baselines.run --alg=ppo2 --env=BreakoutNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/breakout.params --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9
```

Masking is done here:
baselines/baselines/common/trex_utils.py 

Custom reward wrapper is here:
baselines/baselines/common/custom_reward_wrapper.py



## Evaluation of learned policy ##

After training an RL agent use evaluateLearnedPolicy.py to evaluate the performance

For example:

```python evaluateLearnedPolicy.py --env_name breakout --checkpointpath [path_to_rl_checkpoint]```


