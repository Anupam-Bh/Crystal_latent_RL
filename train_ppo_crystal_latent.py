
import os
import argparse
import numpy as np

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from crystal_latent_env import CrystalLatentEnv


class CastObsToFloat32(gym.ObservationWrapper):
    """SB3 expects float32 obs; your env can output float64/float32 depending on code paths."""
    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)


def make_env(
    combined_z_size: int,
    ckpt_name: str,
    mp_api_key: str,
    step_size: float,
    bound: float,
    max_steps: int,
    save_cif: bool,
):
    def _init():
        env = CrystalLatentEnv(
            combined_z_size=combined_z_size,
            ckpt_name=ckpt_name,
            mp_api_key=mp_api_key,
            step_size=step_size,
            bound=bound,
            max_steps=max_steps,   # <-- episode length cap ( env sets done when step_count >= max_steps)
            save_cif=save_cif,
        )
        env = CastObsToFloat32(env)
        env = Monitor(env)  # records episode returns/lengths for SB3 logs
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_crystal_latent")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=1)

    # Env args
    parser.add_argument("--combined_z_size", type=int, default=20)
    parser.add_argument("--ckpt_name", type=str, default="combined_vae.pt")
    parser.add_argument("--mp_api_key", type=str, default="")  # set if needed
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--bound", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=200)  # <-- requested
    parser.add_argument("--save_cif", action="store_true")     # WARNING: heavy I/O; try without first

    # PPO args (sane defaults, tweak as needed)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--n_steps", type=int, default=2048)   # rollout horizon
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)

    # Checkpointing
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    run_dir = os.path.join("runs", args.exp_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Vectorize env
    env_fns = [
        make_env(
            combined_z_size=args.combined_z_size,
            ckpt_name=args.ckpt_name,
            mp_api_key=args.mp_api_key,
            step_size=args.step_size,
            bound=args.bound,
            max_steps=args.max_steps,
            save_cif=args.save_cif,
        )
        for _ in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)  # aggregates episode stats across envs

    # Logger (tensorboard + stdout + csv)
    logger = configure(run_dir, ["stdout", "tensorboard", "csv"])

    # Checkpoint callback: saves model.zip regularly
    checkpoint_cb = CheckpointCallback(
        save_freq=args.ckpt_every,
        save_path=ckpt_dir,
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callbacks = CallbackList([checkpoint_cb])

    # Resume or create new
    latest_path = os.path.join(ckpt_dir, "ppo_latest.zip")
    if args.resume and os.path.exists(latest_path):
        print(f"[train] Resuming from: {latest_path}")
        model = PPO.load(latest_path, env=vec_env, device="auto", print_system_info=True)
        model.set_logger(logger)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            device="auto",
        )
        model.set_logger(logger)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # Final saves
    final_path = os.path.join(run_dir, "ppo_final.zip")
    model.save(final_path)
    model.save(latest_path)  # convenient "resume" pointer

    print(f"[train] Saved final:  {final_path}")
    print(f"[train] Saved latest: {latest_path}")


if __name__ == "__main__":
    main()
