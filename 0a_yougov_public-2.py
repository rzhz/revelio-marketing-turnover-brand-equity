import pandas as pd
import os
import time

sdirectory = '/work/SafeGraph/revelio/data'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'
fdirectory = '/work/SafeGraph/revelio/data/2023latest'

n_job_history = 769

# List to hold the dataframes
dfs = []

for i in range(32):
    # Construct file path
    file_path = os.path.join(sdirectory, f"company_ref_00{str(i).zfill(2)}_part_00.parquet")

    # Read the parquet file
    df = pd.read_parquet(file_path)

    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into one
cik_df = pd.concat(dfs, ignore_index=True)

cik_df = cik_df[cik_df['rcid'].notna()]
cik_df = cik_df[cik_df['cik'].notna()]

for j in range(32,48):
    # Placeholder list to store filtered data
    final_df = pd.DataFrame()
    # Iterate through all the job history CSV files
    for i in range(16*j, 16*(j+1)):
        loop_start_time = time.time()
        file_name = os.path.join(fdirectory, f"Individual_Job_Positions_and_Job_History-{i}.csv.gz")
        if os.path.exists(file_name):
            job_history_df = pd.read_csv(file_name, low_memory=False)
            # Filter for 'United States' in the 'COUNTRY' column
            df = job_history_df[job_history_df['COUNTRY'] == 'United States']
        
            merged_df = df.merge(cik_df, left_on='RCID', right_on='rcid', how='left')
            filtered_df = merged_df[merged_df['cik'].notna()]

            final_df = pd.concat([final_df, filtered_df], ignore_index=True)       
        loop_end_time = time.time()
        print("Time taken for one loop iteration:", loop_end_time - loop_start_time, "seconds")

    final_df.to_csv(os.path.join(tdirectory, f"yougov_public_job_history-{j}.csv"), index=False)
    
    
final_df = pd.DataFrame()
# Iterate through all the job history CSV files
file_name = os.path.join(fdirectory, "Individual_Job_Positions_and_Job_History-768.csv.gz")
if os.path.exists(file_name):
    job_history_df = pd.read_csv(file_name, low_memory=False)
    # Filter for 'United States' in the 'COUNTRY' column
    df = job_history_df[job_history_df['COUNTRY'] == 'United States']
        
    merged_df = df.merge(cik_df, left_on='RCID', right_on='rcid', how='left')
    filtered_df = merged_df[merged_df['cik'].notna()]

    final_df = pd.concat([final_df, filtered_df], ignore_index=True)       
final_df.to_csv(os.path.join(tdirectory, "yougov_public_job_history-48.csv"), index=False)

