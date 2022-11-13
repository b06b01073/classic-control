import gym
from agent import *

env_name = 'CartPole-v1'

def main():
    env = gym.make(env_name, render_mode='human')

    while True:
        agent = Agent(env.action_space.n, env.observation_space.shape[0], train_mode=False)
        agent.local_network.load_state_dict(torch.load(f'{env_name}_weights.pth'))
        agent.local_network.eval()
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
    main()