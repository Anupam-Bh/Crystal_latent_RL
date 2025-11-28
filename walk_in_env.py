from crystal_latent_env import CrystalLatentEnv

env = CrystalLatentEnv(
    combined_z_size=20,
    ckpt_name="combined_vae.pt",
    mp_api_key="YOUR_MP_API_KEY_HERE",  # only if you haven't pre-generated data
    step_size=0.1,
    bound=3.0,
    max_steps=50,
    save_cif=True,
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    # obs is your "measurement plot"; reward is scalar

    print(reward,done,truncated,info['cif_path'], info['formation_energy'])
