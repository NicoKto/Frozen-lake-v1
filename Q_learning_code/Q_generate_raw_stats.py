
import sys
import os

# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_directory)

from Q_learning_code.functions_container import *


parent_directory = f"C:\\-----\\-----\\----"
directory_raw = os.path.join(parent_directory, 'Q_learning_data', 'Q_raw_data')
directory_save_raw_stat = os.path.join(parent_directory, 'Q_learning_data', 'Q_stats')
directory_average = os.path.join(parent_directory, 'Q_learning_data', 'Q_raw_avg')
file_raw_stats = os.path.join(directory_save_raw_stat, 'Q_raw_stats_SF.csv')
file_updated_raw_stats = os.path.join(directory_save_raw_stat, 'Q_raw_stats_updated_SF.csv')
file_consolidated = os.path.join(directory_save_raw_stat, 'Q_consolidated_stats_SF.csv')

sum_col = 'Rewards'
mean_med_cols = ['Steps', 'Difference']  
output_file_name = 'Q_raw_stats_SF.csv'



columns_to_descriptive = {
    'Sum_Rewards': ['min_rewards', 
                    'max_rewards', 
                    'mean_rewards', 
                    'median_rewards', 
                    'iqr_rewards', 
                    'shapiro_rewards'],
    'Steps': ['min_steps', 
              'max_steps', 
              'mean_steps', 
              'median_steps', 
              'iqr_steps', 
              'shapiro_steps'],
    'Difference': ['min_difference', 
                   'max_difference', 
                   'mean_difference', 
                   'median_difference', 
                   'iqr_difference', 
                   'shapiro_difference'],
    'Significant_Increase_Index': ['min_increase_index', 
                                   'max_increase_index', 
                                   'mean_increase_index', 
                                   'median_increase_index', 
                                   'iqr_increase_index', 
                                   'shapiro_increase_index']
}
E_List = [1, 4, 6, 9]
A_List = [1, 4, 6, 9]
G_List = [1, 4, 6, 9]


if __name__ == "__main__":
    df_raw_stats = sum_rewards_in_files(directory_raw, 
                                        directory_save_raw_stat, 
                                        sum_col, 
                                        mean_med_cols, 
                                        output_file_name)
    print(df_raw_stats)
    add_df_to_file(directory_average, file_raw_stats, file_updated_raw_stats)
    
    calculate_and_save_statistics(file_updated_raw_stats, 
                                  columns_to_descriptive, 
                                  file_consolidated, 
                                  E_List, A_List, G_List)
    print(f"Statistics saved to {file_consolidated}")
    
    

