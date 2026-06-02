# Training Tasks Repository

This repository contains training scripts for various manipulation and locomotion tasks using reinforcement learning frameworks (e.g., JAX PPO).  
Each task includes a reference trajectory and corresponding training script.

---

## 🚀 Example Tasks and Training Commands

Often, several tasks share the same environment but use different reference trajectories. For manipulation tasks, you usually need to run the environment "Ref_Tracking_Neckarm_Pickwalkpush_Pick".

### Tri-Manip Shelf Object Relocation
### Neckarm Screwdriving
### Backarm Door Handle Rotation
### Tablemove Manip
### Neckarm Pick
### Neckarm Shelf Manip

```bash
python learning/digit_v3/train_jax_ppo_digit.py   --env_name=<Ref_Tracking_Neckarm_Pickwalkpush_Pick>   --ref_path=<path_to_reference_trajectory>   --num_timesteps=800000000   --num_evals=41   --num_envs=4096   --use_wandb=True   --domain_randomization=True
```

For Locomotion tasks, you usually need to run the environment "Ref_Tracking_Neckarm_Pickwalk_Walk".

### Tablemove Loco
### Neckarm pushdoor

```bash
python learning/digit_v3/train_jax_ppo_digit.py   --env_name=<Ref_Tracking_Neckarm_Pickwalk_Walk>   --ref_path=<path_to_reference_trajectory>   --num_timesteps=800000000   --num_evals=41   --num_envs=4096   --use_wandb=True   --domain_randomization=True
```

for more information, please contact Fukang Liu fukangliu@gatech.edu