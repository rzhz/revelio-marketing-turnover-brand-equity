import pandas as pd
import os
import time
import json

# Define the directory
directory = '/work/SafeGraph/revelio/code'
sdirectory = '/work/SafeGraph/revelio/data'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

n_job_history = 769

# Load brand list data
public_list = pd.read_csv(os.path.join(directory, "public_brand_list_revelio_new.csv"))
public_list['conm'] = public_list['conm'].str.lower()
public_list = public_list.rename(columns={'conm': 'firm'})

dfs = []
# Read and process dataframes
for i in range(32):
    file_path = os.path.join(sdirectory, f"company_ref_00{str(i).zfill(2)}_part_00.parquet")
    df = pd.read_parquet(file_path)
    dfs.append(df)
naics_df = pd.concat(dfs, ignore_index=True)

cik_df = naics_df.dropna(subset=['cik'])
cik_df['cik'] = pd.to_numeric(cik_df['cik'], errors='coerce')
public_list['cik'] = pd.to_numeric(public_list['cik'], errors='coerce')
public_list = pd.merge(public_list, cik_df, left_on=['cik'], right_on=['cik'], how='left')
public_list = public_list.sort_values(by=['cik', 'naics_code', 'firm'])

private_direct = pd.read_csv(os.path.join(directory, "private_brand_rcid_direct_2.csv"))
private_direct['Brand'] = private_direct['Brand'].str.lower()
private_direct = private_direct.rename(columns={'Brand': 'firm'})
private_direct = private_direct.drop_duplicates(subset = ['firm', 'company', 'naics_code', 'rcid'], keep='first')

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

private_fuzzy = pd.concat(dfs, ignore_index=True)
private_fuzzy['Brand'] = private_fuzzy['Brand'].str.lower()
private_fuzzy = private_fuzzy.rename(columns={'Brand': 'firm'})
private_fuzzy = private_fuzzy.drop_duplicates(subset = ['firm', 'company', 'naics_code', 'rcid'], keep='first')

firm_list = pd.concat([public_list, private_direct, private_fuzzy])

def get_peers_and_peers_of_peers(firm_list, naics_df):
    peers_dict = {}
    
    for brand in firm_list['firm'].unique():
        brand_data = firm_list[firm_list['firm'] == brand]
        brand_naics_codes = brand_data['naics_code'].dropna().unique()
        brand_parent_rcid = brand_data['ultimate_parent_rcid'].unique()
        
        # Step 1: Find all rcids with the same first four digits of any of the brand's naics_codes
        potential_peers = naics_df[naics_df['naics_code'].astype(str).str[:4].isin([str(code)[:4] for code in brand_naics_codes])]
        
        # Step 2: Find broad peers (rcids that share the same ultimate parent rcids with these potential peers)
        broad_peers = naics_df[naics_df['ultimate_parent_rcid'].isin(potential_peers['ultimate_parent_rcid'].unique())]
        
        # Step 3: Exclude rcids that have the same ultimate parent rcid as the original brand
        peers = broad_peers[~broad_peers['ultimate_parent_rcid'].isin(brand_parent_rcid)]
        
        peer_naics_codes = peers['naics_code'].dropna().unique()
        peer_parent_rcid = peers['ultimate_parent_rcid'].unique()
        
        # Step 4: Identify peers of peers
        potential_peers_of_peers = naics_df[naics_df['naics_code'].astype(str).str[:4].isin([str(code)[:4] for code in peer_naics_codes])]
        peers_of_peers = naics_df[naics_df['ultimate_parent_rcid'].isin(potential_peers_of_peers['ultimate_parent_rcid'].unique())]
        peers_of_peers = peers_of_peers[~peers_of_peers['ultimate_parent_rcid'].isin(brand_parent_rcid)]
        peers_of_peers = peers_of_peers[~peers_of_peers['ultimate_parent_rcid'].isin(peer_parent_rcid)]
        
        peers_dict[brand] = {
            'peers': peers['rcid'].tolist(),
            'peers_of_peers': peers_of_peers['rcid'].tolist()
        }
    
    return peers_dict

# Get peers and peers of peers
peers_dict = get_peers_and_peers_of_peers(firm_list, naics_df)

# Save peers_dict as a JSON file
with open(os.path.join(directory, "peers_dict.json"), "w") as json_file:
    json.dump(peers_dict, json_file)