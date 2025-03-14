import pandas as pd
import numpy as np
import os

# Paths
directory = '/work/SafeGraph/revelio/code'
sdirectory = '/work/SafeGraph/revelio/data'

# Read the private list
private_list = pd.read_csv(os.path.join(directory, "private_brand_list_updated.csv"))
private_list['Brand'] = private_list['Brand'].str.lower()

# List to hold dataframes
dfs = []

# Read and process dataframes
for i in range(32):
    file_path = os.path.join(sdirectory, f"company_ref_00{str(i).zfill(2)}_part_00.parquet")
    df = pd.read_parquet(file_path)
    df['company'] = df['company'].str.lower()
    df['primary_name'] = df['primary_name'].str.lower()
    df['child_company'] = df['child_company'].str.lower()
    df['ultimate_parent_rcid_name'] = df['ultimate_parent_rcid_name'].str.lower()
    df['company_nospace'] = df['company'].str.lower().str.replace(' ', '')
    df['primary_name_nospace'] = df['primary_name'].str.lower().str.replace(' ', '')
    df['child_company_nospace'] = df['child_company'].str.lower().str.replace(' ', '')
    df['ultimate_parent_rcid_name_nospace'] = df['ultimate_parent_rcid_name'].str.lower().str.replace(' ', '')
    dfs.append(df)

# Concatenate all dataframes
cik_df = pd.concat(dfs, ignore_index=True)

# Direct Matching
direct_matches = cik_df[cik_df['company'].isin(private_list['Brand']) | 
                        cik_df['primary_name'].isin(private_list['Brand'])| 
                        cik_df['child_company'].isin(private_list['Brand'])| 
                        cik_df['ultimate_parent_rcid_name'].isin(private_list['Brand'])|
                        cik_df['company_nospace'].isin(private_list['Brand']) | 
                        cik_df['primary_name_nospace'].isin(private_list['Brand'])| 
                        cik_df['child_company_nospace'].isin(private_list['Brand'])| 
                        cik_df['ultimate_parent_rcid_name_nospace'].isin(private_list['Brand'])]

# Define a function to find the matching brand
def find_matching_brand(row, private_list):
    for column in ['company', 'primary_name', 'child_company', 'ultimate_parent_rcid_name', 'company_nospace', 'primary_name_nospace', 'child_company_nospace', 'ultimate_parent_rcid_name_nospace']:
        if row[column] in private_list['Brand'].values:
            return row[column]
    return np.nan

# Add a 'Brand' column to direct_matches
direct_matches['Brand'] = direct_matches.apply(find_matching_brand, axis=1, private_list=private_list)

direct_matches.to_csv(os.path.join(directory, "private_brand_rcid_direct_updated.csv"), index=False)

# Identifying unmatched entries in private_list
unmatched_brands = private_list[~private_list['Brand'].isin(direct_matches['Brand'])]

# Save to CSV
unmatched_brands.to_csv(os.path.join(directory, "private_unmatched_brands_updated.csv"), index=False)

'''
# Read the private list
private_list = pd.read_csv(os.path.join(directory, "private_brand_list.csv"))
private_list['Brand'] = private_list['Brand'].str.lower().str.replace(' ', '')

# List to hold dataframes
dfs = []

# Read and process dataframes
for i in range(32):
    file_path = os.path.join(sdirectory, f"company_ref_00{str(i).zfill(2)}_part_00.parquet")
    df = pd.read_parquet(file_path)
    df['company_nospace'] = df['company'].str.lower().str.replace(' ', '')
    df['primary_name_nospace'] = df['primary_name'].str.lower().str.replace(' ', '')
    df['child_company_nospace'] = df['child_company'].str.lower().str.replace(' ', '')
    df['ultimate_parent_rcid_name_nospace'] = df['ultimate_parent_rcid_name'].str.lower().str.replace(' ', '')
    dfs.append(df)

# Concatenate all dataframes
cik_df = pd.concat(dfs, ignore_index=True)

# Direct Matching
direct_matches = cik_df[cik_df['company_nospace'].isin(private_list['Brand']) | 
                        cik_df['primary_name_nospace'].isin(private_list['Brand'])| 
                        cik_df['child_company_nospace'].isin(private_list['Brand'])| 
                        cik_df['ultimate_parent_rcid_name_nospace'].isin(private_list['Brand'])]

# Define a function to find the matching brand
def find_matching_brand(row, private_list):
    for column in ['company_nospace', 'primary_name_nospace', 'child_company_nospace', 'ultimate_parent_rcid_name_nospace']:
        if row[column] in private_list['Brand'].values:
            return row[column]
    return np.nan

# Add a 'Brand' column to direct_matches
direct_matches['Brand'] = direct_matches.apply(find_matching_brand, axis=1, private_list=private_list)

direct_matches.to_csv(os.path.join(directory, "private_brand_rcid_direct.csv"), index=False)

# Identifying unmatched entries in private_list
unmatched_brands = private_list[~private_list['Brand'].isin(direct_matches['Brand'])]

# Save to CSV
unmatched_brands.to_csv(os.path.join(directory, "private_unmatched_brands.csv"), index=False)
'''