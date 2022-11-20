import gym
from agent import *
import argparse

env_name = 'CartPole-v1'

def main(algo, env_name):
    env = gym.make(env_name, render_mode='human')

    while True:
        agent = get_agent(algo, env.action_space.n, env.observation_space.shape[0])
        agent.train_mode = False

        agent.policy_newtork.load_state_dict(torch.load(f'{env_name}_weights.pth'))
        agent.policy_newtork.eval()
        obs, _ = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = agent.step(obs)
            obs, reward, termination, truncated, _ = env.step(action)
            total_reward += reward
            if termination or truncated:
                break
        print(total_reward)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', default='DQN')
    parser.add_argument('-e', '--env', default='CartPole-v1')
    args = parser.parse_args()
    algo = args.algo
    env_name = args.env

    main(algo, env_name)