import gymnasium as gym
import random
import numpy as np
import pandas as pd
import os

# Initialize environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

def printQ(Q):
    print("\nQ-TABLE:")
    directions = ["Left", "Down", "Right", "Up"]
    best_actions = []

    # Loop through each state and print Q values for each action
    for state in sorted(Q.keys(), key=lambda x: int(x)):  # Sort states numerically for readability
        best_action_index = np.argmax(Q[state])
        best_action_value = np.max(Q[state])
        action_details = ', '.join([f"{directions[i]}: {round(value, 3)}" for i, value in enumerate(Q[state])])
        print(f"State {state}: {action_details}")
        best_actions.append(f"S{state}:{directions[best_action_index]}")

    # Print summary of best actions all on one line
    print("\nBEST ACTIONS:")
    print(' '.join(best_actions))

def policy(observation, epsilon, Q, env):
    if observation not in Q:
        Q[observation] = np.zeros(env.action_space.n)
    if random.random() < epsilon:
        return random.randint(0, env.action_space.n-1)
    else:
        return np.argmax(Q[observation])

def updateQ(observation, action, reward, next_observation, next_action, alpha, gamma, Q):
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

# Define the parameter ranges for the simulation
epsilon_values = [1]
alpha_values = [0.1, 0.4, 0.6, 0.9]
gamma_values = [0.1, 0.4, 0.6, 0.9]
trials = range(1, 32)
nbr_episodes = 10000
min_epsilon = 0.1

# Define the save path
base_path = f"C:\\-----\\-----\\----"
save_path = os.path.join(base_path, "S_raw_decay_data_1")

# Create the directory if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

for trial in trials:
    for epsilon_initial in epsilon_values:
        for alpha in alpha_values:
            for gamma in gamma_values:
                sarsaEpisodeScore = []
                stepsPerEpisode = []
                episodesOver100Steps = 0
                Q_differences = []

                epsilon = epsilon_initial  # Reset epsilon at the start of each trial
                epsilon_decay = (epsilon_initial - min_epsilon) / nbr_episodes

                observation, info = env.reset()
                observation = str(observation)
                Q = {observation: np.zeros(env.action_space.n)}
                print(Q)
                previous_Q = None

                for episode in range(nbr_episodes):
                    episode_rewards = 0
                    steps = 0

                    action = policy(observation, epsilon, Q, env)  # Initialize action

                    while True:
                        next_observation, reward, terminated, truncated, info = env.step(action)
                        next_observation = str(next_observation)
                        next_action = policy(next_observation, epsilon, Q, env)  # Select next action
                        updateQ(observation, action, reward, next_observation, next_action, alpha, gamma, Q)
                        episode_rewards += reward
                        steps += 1
                        observation, action = next_observation, next_action

                        if terminated or truncated:
                            observation, info = env.reset()
                            observation = str(observation)
                            if observation not in Q:
                                Q[observation] = np.zeros(env.action_space.n)
                            action = policy(observation, epsilon, Q, env)  # Re-initialize action after reset
                            break

                    sarsaEpisodeScore.append(episode_rewards)
                    stepsPerEpisode.append(steps)
                    Q_differences.append(calculate_difference(Q, previous_Q))
                    previous_Q = Q.copy()
                    if steps >= 100:
                        episodesOver100Steps += 1

                    if episode % 100 == 0:
                        sarsaAvg = sum(sarsaEpisodeScore[-100:]) / 100
                        stepsAvg = sum(stepsPerEpisode[-100:]) / 100

                    epsilon = max(min_epsilon, epsilon - epsilon_decay)

                env.close()

                # Print final statistics
                print(f"Total reward: {sum(sarsaEpisodeScore)}")
                print(f"Average number of steps per episode: {sum(stepsPerEpisode) / len(stepsPerEpisode)}")
                print(f"Number of episodes with 100 or more steps: {episodesOver100Steps}")
                printQ(Q)

                # Data preparation
                data = {
                    "Episodes": range(1, nbr_episodes + 1),
                    "Rewards": sarsaEpisodeScore,
                    "Steps": stepsPerEpisode,
                    "Q_Difference": Q_differences
                }

                # Create DataFrame
                df = pd.DataFrame(data)
                # Ensure Rewards are of integer type
                df['Rewards'] = df['Rewards'].astype(int)

                # Save to CSV
                filename = f"E1A{int(alpha*10)}G{int(gamma*10)}T{trial}.csv"
                print(f"saving file {filename}")
                df.to_csv(os.path.join(save_path, filename), index=False)
