import gym
from agent import *
import matplotlib.pyplot as plt
import collections
from statistics import mean
import random
import argparse

env_name = 'LunarLander-v2'

def critic(agent, redo):
    env = gym.make(env_name)
    total_score = 0
    for _ in range(redo):
        obs, _ = env.reset()
        while True:
            action = agent.step(obs)
            obs, reward, termination, truncated, _ = env.step(action)
            total_score += reward
            if termination or truncated:
                break
    return total_score

def random_policy(env, episode):
    total_rewards = []
    rewards_queue = collections.deque(maxlen=100)
    avg_rewards = []
    for _ in range(episode):
        obs = env.reset()
        total_reward = 0
        while True:
            action = random.choice(range(env.action_space.n))
            obs, reward, termination, truncated, _ = env.step(action)
            total_reward += reward
        
            if termination or truncated:
                break
        total_rewards.append(total_reward)
        if len(rewards_queue) >= rewards_queue.maxlen:
            rewards_queue.popleft()
        rewards_queue.append(total_reward)
        avg_rewards.append(mean(rewards_queue))

    return  total_rewards, avg_rewards


def main(algo):

    # init env and agent
    env = gym.make(env_name)
    agent = get_agent(algo, action_dim=env.action_space.n, obs_dim=env.observation_space.shape[0])
    redo = 10

    # training parameters
    episode = 500
    update_step = 20
    step = 0

    # replay buffer
    buffer_capacity = 10000
    replay_buffer = ReplayBuffer(buffer_capacity)

    # record result
    total_rewards = []
    rewards_queue = collections.deque(maxlen=100)
    avg_rewards = []

    # early stop
    # early_stop = 150 # if the model hasn't improve for the previous **early_stop** episodes, we consider the model is not able to improve anymore
    # early_stop_count = 0
    best_episode = float('-inf')
    candidate_score = float('-inf')



    for i in range(episode):
        obs, _ = env.reset()
        total_reward = 0
        while True:
            # env.render()
            step += 1

            action = agent.step(obs)
            next_obs, reward, termination, truncated, _= env.step(action)

            replay_buffer.insert([obs, action, reward, next_obs, termination])
            total_reward += reward

            obs = next_obs


            agent.train(replay_buffer)

            if termination or truncated:
                break
                
            if step % update_step == 0:
                # agent.hard_update()
                agent.soft_update()
        agent.eps_decay()

        print(f'Episode: {i + 1}, Reward: {total_reward}')

        total_rewards.append(total_reward)

        if len(rewards_queue) >= rewards_queue.maxlen:
            rewards_queue.popleft()
        rewards_queue.append(total_reward)
        avg_rewards.append(mean(rewards_queue))

        # In the case of the cartpole game, since the upper bound of total_reward is 500, it will save the last agent's network which enables the agent to reach the reward of 500.
        if total_reward >= best_episode:
            redo_agent = DDQNAgent(action_dim=env.action_space.n, obs_dim=env.observation_space.shape[0], train_mode=False)
            redo_agent.policy_newtork.load_state_dict(agent.policy_newtork.state_dict())
            agent_score = critic(redo_agent, redo)
            if agent_score > candidate_score:
                candidate_score = agent_score
                best_episode = total_reward
                torch.save(agent.policy_newtork.state_dict(), f'{env_name}_weights.pth')
                print('model saved')
            
    
    random_total_rewards, random_avg_rewards = random_policy(env, episode)


    plt.plot(random_total_rewards, label="episodic reward of random policy")
    plt.plot(random_avg_rewards, label="average reward of last 100 episodes of random policy")
    plt.plot(total_rewards, label="episodic reward")
    plt.plot(avg_rewards, label="average reward of last 100 episodes")
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.legend(loc="upper left")
    plt.title(f'{algo} on {env_name}')
    plt.savefig(f'Result_{env_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', default='DQN')
    args = parser.parse_args()
    main(args.algo)