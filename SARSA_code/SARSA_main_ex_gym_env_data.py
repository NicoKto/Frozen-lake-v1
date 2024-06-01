import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def printQ(Q):
    print("\nQ-TABLE:")
    directions = ["Left", "Down", "Right", "Up"]
    best_actions = []

    for state in sorted(Q.keys(), key=lambda x: int(x)):
        best_action_index = np.argmax(Q[state])
        best_action_value = np.max(Q[state])
        action_details = ', '.join([f"{directions[i]}: {round(value, 3)}" for i, value in enumerate(Q[state])])
        print(f"State {state}: {action_details}")
        best_actions.append(f"S{state}:{directions[best_action_index]}")

    print("\nBEST ACTIONS:")
    print(' '.join(best_actions))

def policy(Q, env, observation, epsilon):
    if random.random() > epsilon:
        return random.randint(0, env.action_space.n-1)
    else:
        return np.argmax(Q[observation])

def updateQ(Q, env, observation, action, reward, next_observation, next_action, alpha, gamma):
    if next_observation not in Q:
        Q[next_observation] = np.zeros(env.action_space.n)
    Q[observation][action] = ((1-alpha) * Q[observation][action]) + (alpha * (reward + (gamma * Q[next_observation][next_action])))

def calculate_difference(Q, previous_Q):
    if previous_Q is None:
        return 0
    difference = 0
    for key in Q.keys():
        difference += np.sum(np.abs(Q[key] - previous_Q.get(key, np.zeros_like(Q[key]))))
    return difference

def run_simulation(epsilon, alpha, gamma, trial, nbr_episodes, save_path):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    observation, info = env.reset()
    observation = str(observation)
    Q = {observation: np.zeros(env.action_space.n)}
    previous_Q = None

    sarsaEpisodeScore = []
    sarsaAverageScore = []
    stepsPerEpisode = []
    episodesOver100Steps = 0
    diffEpisode = []

    for episode in range(nbr_episodes):
        episode_rewards = 0
        steps = 0
        
        action = policy(Q, env, observation, epsilon=epsilon)
        
        while True:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = str(next_observation)
            if next_observation not in Q:
                Q[next_observation] = np.zeros(env.action_space.n)
            next_action = policy(Q, env, next_observation, epsilon=epsilon)
            updateQ(Q, env, observation, action, reward, next_observation, next_action, alpha=alpha, gamma=gamma)
            episode_rewards += reward
            steps += 1
            observation, action = next_observation, next_action

            if terminated or truncated:
                diff = calculate_difference(Q, previous_Q)
                diffEpisode.append(diff)
                previous_Q = {k: np.copy(v) for k, v in Q.items()}
                observation, info = env.reset()
                observation = str(observation)
                if observation not in Q:
                    Q[observation] = np.zeros(env.action_space.n)
                action = policy(Q, env, observation, epsilon=epsilon)
                break

        sarsaEpisodeScore.append(episode_rewards)
        stepsPerEpisode.append(steps)
        if steps >= 100:
            episodesOver100Steps += 1

    env.close()

    data = {
        "Episodes": range(1, nbr_episodes + 1),
        "Rewards": sarsaEpisodeScore,
        "Steps": stepsPerEpisode,
        "Difference": diffEpisode
    }

    df = pd.DataFrame(data)
    df['Rewards'] = df['Rewards'].astype(int)
    filename = f"E{int(epsilon*10)}A{int(alpha*10)}G{int(gamma*10)}T{trial}.csv"
    full_path = f"{save_path}/{filename}"
    df.to_csv(full_path, index=False)

    print(f"Total reward: {sum(sarsaEpisodeScore)}")
    print(f"Average number of steps per episode: {sum(stepsPerEpisode) / len(stepsPerEpisode)}")
    print(f"Number of episodes with 100 or more steps: {episodesOver100Steps}")
    printQ(Q)
    print(f"CSV saved as {full_path}")

    return Q, sarsaEpisodeScore, stepsPerEpisode


