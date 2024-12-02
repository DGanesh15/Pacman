import numpy as np
import torch
from time import time
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from tqdm.rich import trange

# List of Atari games
GAMES = [
    'BreakoutNoFrameskip-v4',
    'PongNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4',
    'SeaquestNoFrameskip-v4',
    'QbertNoFrameskip-v4',
    'EnduroNoFrameskip-v4',
    'MsPacmanNoFrameskip-v4',
    'AsteroidsNoFrameskip-v4',
    'BeamRiderNoFrameskip-v4'
]

# Predict function to get actions, activations, and Q-values
def predict(model, observation):
    """
    Get actions, activations, and Q-values from the model.
    """
    model.policy.set_training_mode(False)
    with torch.no_grad():
        obs, _ = model.policy.obs_to_tensor(observation)
        obs = obs.float()
        activations = model.q_net.features_extractor(obs)
        qvalues = model.q_net.q_net(activations)
    actions, _ = model.predict(observation, deterministic=False)
    return actions, activations, qvalues

# Function to generate data from the environment
def generate_data(name, n_steps, seed, **kwargs):
    """
    Generate data from the environment and save it to files.
    """
    
    name = 'MsPacmanNoFrameskip-v4'
    print("DEBUG1: ",name)
    Path(f"data/{name}").mkdir(parents=True, exist_ok=True)
    env = make_atari_env(name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(f'models/dqn-{name}', env=env, custom_objects={
        "learning_rate": 0.1,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    })

    obs = env.reset()
    data = {key: [] for key in kwargs if kwargs[key]}
    data['save_observations'] = []
    data['save_actions'] = []
    

    for _ in trange(n_steps):
        actions, activations, qvalues = predict(model, obs)
        obs, rewards, dones, info = env.step(actions)
        frame = env.render(mode='rgb_array')

        if kwargs.get("save_observations"):
            data['save_observations'].append(obs[0])
        if kwargs.get("save_actions"):
            data['save_actions'].append(actions[0])
        if kwargs.get("save_activations"):
            data['save_activations'].append(activations[0].cpu().numpy())
        if kwargs.get("save_qvalues"):
            data['save_qvalues'].append(qvalues[0].cpu().numpy())
        if kwargs.get("save_rewards"):
            data['save_rewards'].append(rewards[0])
        if kwargs.get("save_images"):
            data['save_images'].append(frame)

    for key, value in data.items():
        np.save(f'data/{name}/{key}_{seed}.npy', np.array(value))

# Function to view gameplay
def view_game(name, n_steps, seed):
    """
    View the game being played by the model.
    """
    env = make_atari_env(name, n_envs=1, seed=seed, env_kwargs={'render_mode': 'human'})
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(f'models/dqn-{name}', env=env, custom_objects={
        "learning_rate": 0.1,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    })

    obs = env.reset()
    for _ in trange(n_steps):
        actions, _, _ = predict(model, obs)
        obs, _, _, _ = env.step(actions)

# Main process handler
def process(games, n_steps, view, seed, **kwargs):
    """
    Process games for data generation or gameplay viewing.
    """
    if view:
        #view_game(games[0], n_steps, seed)
        view_game('MsPacmanNoFrameskip-v4', n_steps, seed)
    else:
        #for game in games:
        generate_data(games, n_steps, seed, **kwargs)

# Command-line argument parser
def main():
    """
    Main function to handle arguments and run the script.
    """
    parser = argparse.ArgumentParser(description="Generate data by running models on Atari games.")
    parser.add_argument('--games', type=str, nargs='+', choices=GAMES,default='[MsPacmanNoFrameskip-v4]',
                        help="List of Atari games to run.")
    parser.add_argument('--n-steps', '-n', type=int, default=10000,
                        help="Number of steps to run the game (default: 10000).")
    parser.add_argument('--view', '-v', action='store_true',
                        help="View gameplay instead of generating data.")
    parser.add_argument('--seed', '-s', type=int, default=int(time()),
                        help="Random seed for environment and model.")
    parser.add_argument('--save-observations', action='store_true', help="Save observations.")
    parser.add_argument('--save-actions', action='store_true', help="Save actions.")
    parser.add_argument('--save-activations', action='store_true', default=True, help="Save activations.")
    parser.add_argument('--save-qvalues', action='store_true', default=True, help="Save Q-values.")
    parser.add_argument('--save-rewards', action='store_true', default=True, help="Save rewards.")
    parser.add_argument('--save-images', action='store_true', default=True, help="Save images.")

    args = parser.parse_args()
    print("DEBUG: ",args.games[0])
    process(
        games=args.games,
        n_steps=args.n_steps,
        view=True,
        seed=args.seed,
        save_observations=args.save_observations,
        save_actions=args.save_actions,
        save_activations=args.save_activations,
        save_qvalues=args.save_qvalues,
        save_rewards=args.save_rewards,
        save_images=args.save_images
    )

if __name__ == '__main__':
    main()
