import gym
from agent import *

env_name = "LunarLander-v2"
file_path = f'videos/{env_name}'


def main():
    env = gym.wrappers.Monitor(gym.make(env_name), file_path, force=True)
    obs = env.reset()
    agent = Agent(env.action_space.n, env.observation_space.shape[0], train_mode=False)
    agent.local_network.load_state_dict(torch.load(f'{env_name}_weights.pth'))
    agent.local_network.eval()
    while True:
        action = agent.step(obs)
        obs, reward, termination, _ = env.step(action)
        if termination:
            break
    env.close()


if __name__ == '__main__':
    main()