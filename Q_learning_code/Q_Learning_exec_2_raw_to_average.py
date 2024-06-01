import pandas as pd
import numpy as np
import os

def process_files(source_path: str, destination_path: str, episodes_per_group: int = 100) -> None:
    # Check if destination path exists, create if not
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    
    # List all csv files in the source directory
    for filename in os.listdir(source_path):
        if filename.endswith('.csv'):
            # Load the CSV file
            file_path = os.path.join(source_path, filename)
            df = pd.read_csv(file_path)
            
            # Group the data by each 'episodes_per_group' episodes
            grouped = df.groupby(df.index // episodes_per_group)
            
            # Creating summary DataFrame
            df_summary = pd.DataFrame({
                "Episodes_100": np.arange(episodes_per_group, len(df) + episodes_per_group, episodes_per_group),
                "Sum_Rewards": grouped["Rewards"].sum(),
                "Avg_Rewards": grouped["Rewards"].mean(),
                "Avg_Steps": grouped["Steps"].mean(),
                "Avg_Difference": grouped["Difference"].mean()
            })

            df_summary.reset_index(drop=True, inplace=True)
            
            # Save the summarized DataFrame to a new CSV file
            output_filename = 'AV' + filename
            output_path = os.path.join(destination_path, output_filename)
            df_summary.to_csv(output_path, index=False)
            
            print(f'Processed and saved: {output_filename}')


source_path = f"C:\\-----\\-----\\----"
destination_path =  f"C:\\-----\\-----\\----"
process_files(source_path, destination_path)
