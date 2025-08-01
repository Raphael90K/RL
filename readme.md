# Vertiefung Künstliche Intelligenz: Reinforcement Learning
Sommersemester 2025

Curiostiy-driven Reinforcement Learning with RND, ICM und BYOL-explore

## Project Structure

<pre> 
src/ 
├── config.py 
├── train_parallel.py
├── callbacks/ 
│ ├── logPlainRewardCallback.py 
│ ├── logRewardCallback.py 
│ └── uniquePositionCallback.py 
├── envs/ 
│ ├── action_wrapper.py 
│ ├── env.py 
│ ├── observation_wrapper.py 
│ └── reward_wrapper.py 
└── intrinsic/ 
  ├── byol_model.py 
  ├── icm_model.py 
  └── rnd_model.py 
 </pre>