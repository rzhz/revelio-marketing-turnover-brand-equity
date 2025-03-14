import pandas as pd
import numpy as np
import os

# Define the directory containing the files
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

public_list = pd.read_csv(os.path.join(directory, "public_brand_list.csv"))
public_list['conm'] = public_list['conm'].str.lower()

# Placeholder list to store filtered data
job_history = pd.DataFrame()

# Iterate through all the job history CSV files
for i in range(0,49):
    file_name = os.path.join(tdirectory, f"yougov_public_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False, lineterminator='\n')
        
        merged_df = df.merge(public_list, on='cik', how='left')
        merged_df = merged_df[merged_df['conm'].notna()]
        merged_df = merged_df.rename(columns={'conm': 'firm'})

        job_history = pd.concat([job_history, merged_df], ignore_index=True)       

# Select subset of public_list whose cik appears in job_history
subset_public_list = public_list[public_list['cik'].isin(job_history['cik'].unique())]

subset_public_list.to_csv(os.path.join(directory, "public_brand_list_revelio_new.csv"), index=False)