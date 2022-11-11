import gym
from agent import *

env_name = 'LunarLander-v2'

def main():
    env = gym.make(env_name)

    while True:
        agent = Agent(env.action_space.n, env.observation_space.shape[0], train_mode=False)
        agent.local_network.load_state_dict(torch.load(f'{env_name}_weights.pth'))
        agent.local_network.eval()
        obs = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = agent.step(obs)
            obs, reward, termination, _ = env.step(action)
            total_reward += reward
            if termination:
                break
        print(total_reward)
    env.close()


if __name__ == '__main__':
    main()