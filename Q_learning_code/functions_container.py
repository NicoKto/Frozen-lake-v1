import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, anderson
import ruptures as rpt
import scipy.stats as stats
from scipy import stats
from typing import Optional, Union, List, Dict




def process_column(data: pd.DataFrame, column: str) -> Optional[Union[int, float]]:
    # Filter out NaN and infinite values
    valid_data = data[column].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) == 0:
        return None
    
    # Check if the entire distribution is zero
    if valid_data.nunique() == 1 and valid_data.unique()[0] == 0:
        return 0
    
    # Choose the normality test based on the number of samples
    if len(valid_data) > 5000:
        # Use Anderson-Darling test for large sample sizes
        result = anderson(valid_data)
        # The critical value at the significance level of 0.05
        is_normal = result.statistic < result.critical_values[2]
    else:
        # Use Shapiro-Wilk test for smaller sample sizes
        stat, p_value = shapiro(valid_data)
        alpha = 0.05
        is_normal = p_value > alpha
    
    if is_normal:
        # Distribution is normal
        result = valid_data.mean()
    else:
        # Distribution is not normal
        result = valid_data.median()
    
    return result

def sum_rewards_in_files(directory_path: str, save_directory: str, sum_col: str, mean_med_cols: List[str], output_file_name: str) -> pd.DataFrame:
    # List to store the filename, sum of rewards, and statistics for other columns
    results = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Sum the specified column
                sum_value = df[sum_col].sum()
                # Process the additional columns
                stats = {}
                for col in mean_med_cols:
                    if col in df.columns:
                        stats[f'{col}'] = process_column(df, col)
                    else:
                        stats[f'Mean_Med_{col}'] = None
                # Append the result to the list
                result = {'Filename': filename[:-4], f'Sum_{sum_col}': sum_value}
                result.update(stats)
                results.append(result)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(results)

    # Path to save the resulting CSV file
    save_path = os.path.join(save_directory, output_file_name)

    # Save the DataFrame as a CSV file
    result_df.to_csv(save_path, index=False)

    return result_df

def detect_significant_increase(data: pd.DataFrame, column: str) -> Optional[int]:
    # Extract the specified column as a numpy array
    series = data[column].values

    # Perform change point detection
    model = "l2"  # Model for change point detection
    algo = rpt.Pelt(model=model, min_size=3, jump=1).fit(series)
    result = algo.predict(pen=10)  # Penalty term, to be adjusted based on the dataset

    # The result contains the indices of the change points
    if len(result) > 1:
        significant_increase_index = result[0]
    else:
        significant_increase_index = None
    
    return significant_increase_index

def add_significant_increase_info(df: pd.DataFrame, repository_path: str) -> pd.DataFrame:
    # List to store the significant increase information
    significant_increase_info = []

    for filename in df['Filename']:
        modified_filename = f"AV{filename}.csv"
        file_path = os.path.join(repository_path, modified_filename)
        if os.path.isfile(file_path):
            try:
                # Read the corresponding CSV file
                data = pd.read_csv(file_path)
                # Detect the significant increase in the 'Sum_Rewards' column
                significant_increase_index = detect_significant_increase(data, 'Sum_Rewards')
                significant_increase_info.append(significant_increase_index)
            except Exception as e:
                print(f"Error processing file {modified_filename}: {e}")
                significant_increase_info.append(None)
        else:
            print(f"File {modified_filename} not found in the repository.")
            significant_increase_info.append(None)
    
    df['Significant_Increase_Index'] = significant_increase_info
    print(df)
    return df

def add_df_to_file(repository_path: str, input_file: str, output_file: str) -> None:

    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Add the significant increase information
    df = add_significant_increase_info(df, repository_path)

    # Save the updated DataFrame to the new CSV file
    df.to_csv(output_file, index=False)

    print(df)






def parameters_descritive_stats(df: pd.DataFrame, identifier: str, columns: Dict[str, List[str]]) -> Dict[str, Dict[str, Union[float, None]]]:
    # Filter rows based on identifier in the Filename column
    filtered_df = df[df['Filename'].str.contains(identifier)]
    
    # Initialize a dictionary to hold the results
    results = {}
    
    for column, stats_names in columns.items():
        if len(stats_names) != 6:
            raise ValueError(f"Expected 6 statistics names for column '{column}', but got {len(stats_names)}")
        
        # Exclude rows where the specified column has NaN values
        column_filtered_df = filtered_df.dropna(subset=[column])
        
        # Calculate the required statistical measures
        min_value = column_filtered_df[column].min()
        max_value = column_filtered_df[column].max()
        mean_value = column_filtered_df[column].mean()
        median_value = column_filtered_df[column].median()
        iqr_value = column_filtered_df[column].quantile(0.75) - column_filtered_df[column].quantile(0.25)
        
        # Check if there are at least 3 data points and if the range of the data is non-zero
        if len(column_filtered_df) < 3 or min_value == max_value:
            shapiro_p_value = None  # Shapiro-Wilk test is not valid
        else:
            shapiro_p_value = stats.shapiro(column_filtered_df[column])[1]  # Get p-value from Shapiro-Wilk test
        
        results[column] = {
            stats_names[0]: min_value,
            stats_names[1]: max_value,
            stats_names[2]: mean_value,
            stats_names[3]: median_value,
            stats_names[4]: iqr_value,
            stats_names[5]: shapiro_p_value
        }
    
    return results

def calculate_and_save_statistics(file_path: str, columns_to_columns: Dict[str, List[str]], file_to_save: str, E_List: List[int], A_List: List[int], G_List: List[int]) -> None:
    # Read the file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Initialize a list to hold all results
    all_results = []
    
    for E in E_List:
        for A in A_List:
            for G in G_List:
                identifier = f'E{E}A{A}G{G}'
                stats_result = parameters_descritive_stats(df, identifier, columns_to_columns)
                combined_result = {'Parameters': identifier}
                for col, values in stats_result.items():
                    combined_result.update(values)
                all_results.append(combined_result)
    
    # Convert all results to a DataFrame
    final_df = pd.DataFrame(all_results)
    
    # Save to CSV
    final_df.to_csv(file_to_save, index=False)

    
    

