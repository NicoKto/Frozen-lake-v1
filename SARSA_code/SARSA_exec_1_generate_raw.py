import os
import datetime
from SARSA_main_ex_gym_env_data import run_simulation  # Assuming this import is correct

# Define the parameter ranges for the simulation
epsilon_values = [0.1, 0.4, 0.6, 0.9]
alpha_values = [0.1, 0.4, 0.6, 0.9]
gamma_values = [0.1, 0.4, 0.6, 0.9]
trials = range(1, 41)  
nbr_episodes = 10000
save_path = f"C:\\-----\\-----\\----"

# Initialize the overall process start time
overall_start_time = datetime.datetime.now()
print("Starting all simulations at:", overall_start_time)

# Iterate over all combinations of epsilon, alpha, gamma, and trials
for eps in epsilon_values:
    for alpha in alpha_values:
        for gamma in gamma_values:
            # Ensure the directory exists
            os.makedirs(save_path, exist_ok=True)
            for trial in trials:
                simulation_start_time = datetime.datetime.now()
                try:
                    Q, episode_scores, steps_per_episode = run_simulation(eps, 
                                                                          alpha, 
                                                                          gamma, 
                                                                          trial, 
                                                                          nbr_episodes, 
                                                                          save_path)
                    simulation_end_time = datetime.datetime.now()
                    simulation_duration = simulation_end_time - simulation_start_time
                    print(f"Simulation for epsilon={eps}, alpha={alpha}, gamma={gamma}, trial={trial} complete.")
                    print(f"Start time: {simulation_start_time}")
                    print(f"End time: {simulation_end_time}")
                    print(f"Duration: {simulation_duration}")
                except Exception as e:
                    print(f"Failed to save data for epsilon={eps}, alpha={alpha}, gamma={gamma}, trial={trial}: {e}")

overall_end_time = datetime.datetime.now()
total_duration = overall_end_time - overall_start_time
print("All simulations completed.")
print("Total duration for all simulations:", total_duration)
