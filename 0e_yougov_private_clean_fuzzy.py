import pandas as pd
import os
import time

directory = '/work/SafeGraph/revelio/code'
fdirectory = '/work/SafeGraph/revelio/data/2023latest'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

n_job_history = 769

# List to hold dataframes
dfs = []

# Read and process dataframes
for i in range(0, 12):
    file_path = os.path.join(tdirectory, f"private_brand_rcid_fuzzy_manual-{i}.csv")
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp1252')
    dfs.append(df)

# Concatenate all dataframes
private_list = pd.concat(dfs, ignore_index=True)

for j in range(0,3):
    # Placeholder list to store filtered data
    final_df = pd.DataFrame()
    # Iterate through all the job history CSV files
    for i in range(200*j, 200*(j+1)):
        loop_start_time = time.time()
        file_name = os.path.join(fdirectory, f"Individual_Job_Positions_and_Job_History-{i}.csv.gz")
        if os.path.exists(file_name):
            job_history_df = pd.read_csv(file_name, low_memory=False)
            # Filter for 'United States' in the 'COUNTRY' column
            df = job_history_df[job_history_df['COUNTRY'] == 'United States']
        
            merged_df = df.merge(private_list, left_on='RCID', right_on='rcid', how='inner')

            final_df = pd.concat([final_df, merged_df], ignore_index=True)       
        loop_end_time = time.time()
        print("Time taken for one loop iteration:", loop_end_time - loop_start_time, "seconds")

    final_df.to_csv(os.path.join(tdirectory, f"yougov_private_fuzzy_job_history-{j}.csv"), index=False)

j = 3
# Placeholder list to store filtered data
final_df = pd.DataFrame()
# Iterate through all the job history CSV files
for i in range(200*j, n_job_history):
    loop_start_time = time.time()
    file_name = os.path.join(fdirectory, f"Individual_Job_Positions_and_Job_History-{i}.csv.gz")
    if os.path.exists(file_name):
        job_history_df = pd.read_csv(file_name, low_memory=False)
        # Filter for 'United States' in the 'COUNTRY' column
        df = job_history_df[job_history_df['COUNTRY'] == 'United States']
        
        merged_df = df.merge(private_list, left_on='RCID', right_on='rcid', how='inner')

        final_df = pd.concat([final_df, merged_df], ignore_index=True)       
    loop_end_time = time.time()
    print("Time taken for one loop iteration:", loop_end_time - loop_start_time, "seconds")

final_df.to_csv(os.path.join(tdirectory, f"yougov_private_fuzzy_job_history-3.csv"), index=False)

'''
directory = '/work/SafeGraph/revelio/code'
sdirectory = '/work/SafeGraph/revelio/data'
tdirectory = '/work/SafeGraph/revelio/data/yougov'


n_job_history = 2353
n_skills = 1876
n_user = 1049
n_education = 4

try:
    private_list = pd.read_csv(os.path.join(directory, "private_brand_rcid_fuzzy_manual.csv"), encoding='latin1')
except UnicodeDecodeError:
    try:
        private_list = pd.read_csv(os.path.join(directory, "private_brand_rcid_fuzzy_manual.csv"), encoding='ISO-8859-1')
    except UnicodeDecodeError:
        private_list = pd.read_csv(os.path.join(directory, "private_brand_rcid_fuzzy_manual.csv"), encoding='cp1252')
        
# Placeholder list to store filtered data
final_df = pd.DataFrame()
# Iterate through all the job history CSV files
for i in range(n_job_history):
    loop_start_time = time.time()
    file_name = os.path.join(sdirectory, f"cleaned_individual_job_positions_and_job_history_file-{i}.csv")
    if os.path.exists(file_name):
        job_history_df = pd.read_csv(file_name, low_memory=False)
        # Filter for 'United States' in the 'COUNTRY' column
        df = job_history_df[job_history_df['COUNTRY'] == 'United States']
    
        merged_df = df.merge(private_list, left_on='RCID', right_on='rcid', how='inner')

        final_df = pd.concat([final_df, merged_df], ignore_index=True)       
    loop_end_time = time.time()
    print("Time taken for one loop iteration:", loop_end_time - loop_start_time, "seconds")

final_df.to_csv(os.path.join(tdirectory, f"yougov_private_fuzzy_job_history.csv"), index=False)
'''
