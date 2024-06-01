import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from typing import Dict, List, Union , Optional, Tuple

def printQ(Q: Dict[int, List[float]]) -> None:
    # Print the Q-table and determine the best actions for each state
    print("\nQ-TABLE:")
    directions = ["Left", "Down", "Right", "Up"]
    best_actions = []

    # Iterate over each state in the Q-table, sorted by state number
    for state in sorted(Q.keys(), key=lambda x: int(x)):
        # Determine the best action for the current state
        best_action_index = np.argmax(Q[state])
        best_action_value = np.max(Q[state])
        # Format the Q-values for each direction
        action_details = ', '.join([f"{directions[i]}: {round(value, 3)}" for i, value in enumerate(Q[state])])
        # Print the Q-values for the current state
        print(f"State {state}: {action_details}")
        # Append the best action for the current state to the list
        best_actions.append(f"S{state}:{directions[best_action_index]}")

    # Print the best actions for all states
    print("\nBEST ACTIONS:")
    print(' '.join(best_actions))
    
def policy(Q: Dict[int, List[float]], env, observation: int, epsilon: float) -> int:
    # Epsilon-greedy policy: choose a random action with probability epsilon, otherwise choose the best action
    if random.random() > epsilon:
        # Choose a random action
        return random.randint(0, env.action_space.n - 1)
    else:
        # Choose the best action based on the Q-table
        return np.argmax(Q[observation])

def updateQ(Q: Dict[int, np.ndarray], env, observation: int, action: int, reward: float, next_observation: int, alpha: float, gamma: float) -> None:
    # Ensure the next observation state exists in the Q-table
    if next_observation not in Q:
        Q[next_observation] = np.zeros(env.action_space.n)
    
    # Update the Q-value for the given state and action using the Q-learning formula
    Q[observation][action] = ((1 - alpha) * Q[observation][action]) + \
                             (alpha * (reward + (gamma * np.max(Q[next_observation]))))
def calculate_difference(Q: Dict[int, np.ndarray], previous_Q: Optional[Dict[int, np.ndarray]]) -> float:
    # Calculate the total difference between the current Q-table and the previous Q-table
    if previous_Q is None:
        return 0.0
    
    difference = 0.0
    # Iterate over each state in the current Q-table
    for key in Q.keys():
        # Compute the absolute difference for the current state and accumulate it
        difference += np.sum(np.abs(Q[key] - previous_Q.get(key, np.zeros_like(Q[key]))))
    
    return difference




def run_simulation(epsilon: float, alpha: float, gamma: float, trial: int, nbr_episodes: int, save_path: str) -> Tuple[Dict[str, np.ndarray], List[int], List[int]]:
    # Initialize the FrozenLake environment
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    observation, info = env.reset()
    observation = str(observation)
    
    # Initialize the Q-table and previous Q-table
    Q = {observation: np.zeros(env.action_space.n)}
    previous_Q = None

    # Lists to store metrics for each episode
    qlearningEpisodeScore = []
    qlearningAverageScore = []
    stepsPerEpisode = []
    episodesOver100Steps = 0
    diffEpisode = []

    for episode in range(nbr_episodes):
        episode_rewards = 0
        steps = 0
        
        while True:
            # Choose an action based on the policy
            action = policy(Q, env, observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = str(next_observation)
            
            # Update the Q-table
            updateQ(Q, env, observation, action, reward, next_observation, alpha=alpha, gamma=gamma)
            
            episode_rewards += reward
            steps += 1
            observation = next_observation

            if terminated or truncated:
                # Calculate the difference between the current and previous Q-table
                diff = calculate_difference(Q, previous_Q)
                diffEpisode.append(diff)
                
                # Store a copy of the current Q-table as the previous Q-table
                previous_Q = {k: np.copy(v) for k, v in Q.items()}
                
                # Reset the environment
                observation, info = env.reset()
                observation = str(observation)
                if observation not in Q:
                    Q[observation] = np.zeros(env.action_space.n)
                break

        # Record metrics for the current episode
        qlearningEpisodeScore.append(episode_rewards)
        stepsPerEpisode.append(steps)
        if steps >= 100:
            episodesOver100Steps += 1

    env.close()

    # Create a DataFrame to save episode data
    data = {
        "Episodes": range(1, nbr_episodes + 1),
        "Rewards": qlearningEpisodeScore,
        "Steps": stepsPerEpisode,
        "Difference": diffEpisode
    }

    df = pd.DataFrame(data)
    df['Rewards'] = df['Rewards'].astype(int)
    
    # Construct the filename and save the DataFrame as a CSV file
    filename = f"E{int(epsilon*10)}A{int(alpha*10)}G{int(gamma*10)}T{trial}.csv"
    full_path = f"{save_path}/{filename}"
    df.to_csv(full_path, index=False)

    # Print summary statistics and the Q-table
    print(f"Total reward: {sum(qlearningEpisodeScore)}")
    print(f"Average number of steps per episode: {sum(stepsPerEpisode) / len(stepsPerEpisode)}")
    print(f"Number of episodes with 100 or more steps: {episodesOver100Steps}")
    printQ(Q)
    print(f"CSV saved as {full_path}")

    return Q, qlearningEpisodeScore, stepsPerEpisode
