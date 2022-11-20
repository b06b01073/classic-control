import gym
from agent import *
import argparse


# unfortunately gym doesn't support Monitor wrapper anymore, you can downgrade gym version to run this file
def main(algo, env_name):
    file_path = f'videos/{env_name}'
    env = gym.wrappers.Monitor(gym.make(env_name), file_path, force=True)
    obs = env.reset()
    agent = get_agent(algo, env.action_space.n, env.observation_space.shape[0])
    agent.train_mode = False

    agent.policy_newtork.load_state_dict(torch.load(f'{env_name}_weights.pth'))
    agent.policy_newtork.eval()
    while True:
        action = agent.step(obs)
        obs, reward, termination, truncated, _ = env.step(action)
        if termination or truncated: 
            break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', default='DQN')
    parser.add_argument('-e', '--env', default='CartPole-v1')
    args = parser.parse_args()
    algo = args.algo
    env_name = args.env

    main(algo, env_name)