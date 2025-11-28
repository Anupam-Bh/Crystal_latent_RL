# Crystal Latent-Space RL 


<img width="2562" height="1329" alt="Pipeline1 drawio" src="https://github.com/user-attachments/assets/1257030a-1309-4adf-8a5e-7a4755f865c8" />

This repo contains a Gymnasium environment (`CrystalLatentEnv`) for exploring the **Combined-VAE latent space** from the Crystal-LSBO architecture, plus two ways to explore it:

1. **Random latent walk** (`walk_in_env.py`)
2. **Reinforcement Learning** using **Stable-Baselines3 PPO** (`train_ppo_sb3.py`)

The environment decodes a latent vector `z → FTCP → (optional CIF/POSCAR)` and computes rewards using:
- an XGBoost formation energy predictor (black-box surrogate),
- optional additional physics/ML reward terms (e.g., Fairchem adsorption energy + CV/VAE overpotential pipeline).

---

## Files

- `crystal_latent_env.py`  
  Gymnasium environment class `CrystalLatentEnv` operating in latent space.

- `walk_in_env.py`  
  Simple random exploration in latent space (sample random actions and step).

- `train_ppo_sb3.py`  
  PPO training script using Stable-Baselines3, with checkpoint saving and TensorBoard logs.

---

## Environment overview

### State and actions
- **Internal state**: latent vector `z` of dimension `combined_z_size`.
- **Action space**: `Discrete(2 * combined_z_size)`  
  Each dimension has two actions: `+step_size` or `-step_size`.

### Observations
- Default observation is a flattened slice of the decoded FTCP tensor:
  - FTCP slice `(185, 4)` → flattened to `740`-dim vector.

### Episode length
The environment terminates when:
- `step_count >= max_steps`

Set `max_steps=200` to enforce 200-step episodes.

---

## Installation
### Recommended Python environment
Use Python 3.9+ and install dependencies:
install gymnasium stable-baselines3 torch numpy pandas pymatgen xgboost joblib


### First  in working_VAE_CV   run data_prep_cv_vae_pred_head.py  : It will create VAE checkpoints for measurement analysis
### Then run walk_in_env  : it will generate the FTCP representation bounds from Materials Project and then it will explore the latent space. 
### Files in the 'data' and 'crystal_lsbo_repo' folder are data-files and model checkpoints which can be obtained from Crystal-LSBO repo (https://github.com/onurboyar/Crystal-LSBO)


