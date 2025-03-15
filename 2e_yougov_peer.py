import pandas as pd
import os
import json
import time

# Define the directory
directory = '/work/SafeGraph/revelio/code'

# Load peers_dict from a JSON file
with open(os.path.join(directory, "peers_dict.json"), "r") as json_file:
    peers_dict = json.load(json_file)

# Load job history data
df = pd.read_csv(os.path.join(directory, "yougov_mkt_peer_job_history.csv"), lineterminator='\n', low_memory=False)

# Combine seniority levels
df['SENIORITY_LEVEL'] = df['SENIORITY_LEVEL'].replace({1: 1, 2: 1, 4: 4, 5: 5})

# Convert dates to datetime once for the whole dataset
df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')

# Sort the entire dataset once for efficiency
df.sort_values(by=['USER_ID', 'STARTDATE'], inplace=True)

# Precompute lagged column for 'previous_parent' globally
df['previous_parent'] = df.groupby('USER_ID')['ULTIMATE_PARENT_RCID'].shift(1)

# Initialize aggregated dataframes
aggregated_peer_hire = []
aggregated_peer_leave = []
aggregated_peer_peer_hire = []
aggregated_peer_peer_leave = []

# Iterate through brands in peers_dict
for brand, peers_info in peers_dict.items():
    loop_start_time = time.time()

    # Extract peer and peers-of-peers RCIDs
    peers_rcids = set(peers_info['peers'])
    peers_of_peers_rcids = set(peers_info['peers_of_peers'])

    # Filter data once for peers and peers of peers
    peer_df = df[df['RCID'].isin(peers_rcids)]
    peer_peer_df = df[df['RCID'].isin(peers_of_peers_rcids)]

    # Filter rows where the brand has changed or previous brand is missing
    filtered_peer_df = peer_df[(peer_df['ULTIMATE_PARENT_RCID'] != peer_df['previous_parent']) | peer_df['previous_parent'].isna()]
    filtered_peer_peer_df = peer_peer_df[(peer_peer_df['ULTIMATE_PARENT_RCID'] != peer_peer_df['previous_parent']) | peer_peer_df['previous_parent'].isna()]

    # Precompute unique ULTIMATE_PARENT_RCID counts for peers and peers of peers
    num_peers = filtered_peer_df['ULTIMATE_PARENT_RCID'].nunique()
    num_peers_of_peers = filtered_peer_peer_df['ULTIMATE_PARENT_RCID'].nunique()

    # Process seniority levels
    for seniority in [1, 3, 4, 5]:
        # Filter and aggregate for peers
        peer_hires = (
            filtered_peer_df[filtered_peer_df['SENIORITY_LEVEL'] == seniority]
            .groupby(['STARTDATE'])
            .size()
            .reset_index(name=f'peer_hire_count_{seniority}')
        )
        peer_leaves = (
            filtered_peer_df[filtered_peer_df['SENIORITY_LEVEL'] == seniority]
            .groupby(['ENDDATE'])
            .size()
            .reset_index(name=f'peer_leave_count_{seniority}')
        )
        peer_hires['firm'] = brand
        peer_leaves['firm'] = brand

        # Add average values
        peer_hires[f'avg_peer_hires_{seniority}'] = peer_hires[f'peer_hire_count_{seniority}'] / num_peers
        peer_leaves[f'avg_peer_leaves_{seniority}'] = peer_leaves[f'peer_leave_count_{seniority}'] / num_peers

        # Filter and aggregate for peers-of-peers
        peer_peer_hires = (
            filtered_peer_peer_df[filtered_peer_peer_df['SENIORITY_LEVEL'] == seniority]
            .groupby(['STARTDATE'])
            .size()
            .reset_index(name=f'peer_peer_hire_count_{seniority}')
        )
        peer_peer_leaves = (
            filtered_peer_peer_df[filtered_peer_peer_df['SENIORITY_LEVEL'] == seniority]
            .groupby(['ENDDATE'])
            .size()
            .reset_index(name=f'peer_peer_leave_count_{seniority}')
        )
        peer_peer_hires['firm'] = brand
        peer_peer_leaves['firm'] = brand

        # Add average values
        peer_peer_hires[f'avg_peer_peer_hires_{seniority}'] = peer_peer_hires[f'peer_peer_hire_count_{seniority}'] / num_peers_of_peers
        peer_peer_leaves[f'avg_peer_peer_leaves_{seniority}'] = peer_peer_leaves[f'peer_peer_leave_count_{seniority}'] / num_peers_of_peers

        # Append results to lists
        aggregated_peer_hire.append(peer_hires)
        aggregated_peer_leave.append(peer_leaves)
        aggregated_peer_peer_hire.append(peer_peer_hires)
        aggregated_peer_peer_leave.append(peer_peer_leaves)

    loop_end_time = time.time()
    print(f"Time taken for processing brand {brand}: {loop_end_time - loop_start_time:.2f} seconds")

# Concatenate aggregated results
aggregated_peer_hire = pd.concat(aggregated_peer_hire, ignore_index=True)
aggregated_peer_leave = pd.concat(aggregated_peer_leave, ignore_index=True)
aggregated_peer_peer_hire = pd.concat(aggregated_peer_peer_hire, ignore_index=True)
aggregated_peer_peer_leave = pd.concat(aggregated_peer_peer_leave, ignore_index=True)

# Add `num_peers` and `num_peers_of_peers` columns to each DataFrame
aggregated_peer_hire['num_peers'] = num_peers
aggregated_peer_leave['num_peers'] = num_peers
aggregated_peer_peer_hire['num_peers_of_peers'] = num_peers_of_peers
aggregated_peer_peer_leave['num_peers_of_peers'] = num_peers_of_peers

# Save aggregated data
aggregated_peer_hire.to_csv(os.path.join(directory, 'aggregated_peer_hire.csv'), index=False)
aggregated_peer_leave.to_csv(os.path.join(directory, 'aggregated_peer_leave.csv'), index=False)
aggregated_peer_peer_hire.to_csv(os.path.join(directory, 'aggregated_peer_peer_hire.csv'), index=False)
aggregated_peer_peer_leave.to_csv(os.path.join(directory, 'aggregated_peer_peer_leave.csv'), index=False)

'''
import pandas as pd
import os
import re
import time
import json

# Define the directory containing the files and the pattern of filenames to search for
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

# Load peers_dict from a JSON file
with open(os.path.join(directory, "peers_dict.json"), "r") as json_file:
    peers_dict = json.load(json_file)

# Aggregate data for peers
aggregated_peer_hire = pd.DataFrame()
aggregated_peer_leave = pd.DataFrame()
aggregated_peer_peer_hire = pd.DataFrame()
aggregated_peer_peer_leave = pd.DataFrame()

# Filter all job history data using the collected RCIDs
filtered_job_history = pd.read_csv(os.path.join(directory, "yougov_mkt_peer_job_history.csv"), lineterminator='\n', low_memory=False)
# Now process each brand using the pre-filtered job history data
summary_records = []

for brand, peers_info in peers_dict.items():
    loop_start_time = time.time()
    peers_rcids = peers_info['peers']
    peers_of_peers_rcids = peers_info['peers_of_peers']

    # Filter job history for peers and peers of peers
    peer_df = filtered_job_history[filtered_job_history['RCID'].isin(peers_rcids)]
    peer_peer_df = filtered_job_history[filtered_job_history['RCID'].isin(peers_of_peers_rcids)]

    # Sort by USER_ID and STARTDATE
    peer_df['STARTDATE'] = pd.to_datetime(peer_df['STARTDATE'], errors='coerce')
    peer_df['ENDDATE'] = pd.to_datetime(peer_df['ENDDATE'], errors='coerce')
    peer_peer_df['STARTDATE'] = pd.to_datetime(peer_peer_df['STARTDATE'], errors='coerce')
    peer_peer_df['ENDDATE'] = pd.to_datetime(peer_peer_df['ENDDATE'], errors='coerce')
    
    peer_df = peer_df.sort_values(by=['USER_ID', 'STARTDATE'])
    peer_peer_df = peer_peer_df.sort_values(by=['USER_ID', 'STARTDATE'])

    # Create a lagged column for the previous brand
    peer_df['previous_parent'] = peer_df.groupby('USER_ID')['ULTIMATE_PARENT_RCID'].shift(1)
    peer_peer_df['previous_parent'] = peer_peer_df.groupby('USER_ID')['ULTIMATE_PARENT_RCID'].shift(1)

    # Filter to keep only rows where the brand has changed or previous brand is missing
    filtered_peer_df = peer_df[(peer_df['ULTIMATE_PARENT_RCID'] != peer_df['previous_parent']) | peer_df['previous_parent'].isna()]
    filtered_peer_peer_df = peer_peer_df[(peer_peer_df['ULTIMATE_PARENT_RCID'] != peer_peer_df['previous_parent']) | peer_peer_df['previous_parent'].isna()]

    # Initialize dictionaries to hold aggregated data for each SENIORITY level
    peer_hires_dict = {}
    peer_leaves_dict = {}
    peer_peer_hires_dict = {}
    peer_peer_leaves_dict = {}

    # Aggregate data for each SENIORITY level
    for seniority in [1, 2, 3, 4, 5]:
        peer_hires = filtered_peer_df[filtered_peer_df['SENIORITY_LEVEL'] == seniority].groupby(['STARTDATE']).size().reset_index(name=f'peer_hire_count_{seniority}')
        peer_leaves = filtered_peer_df[filtered_peer_df['SENIORITY_LEVEL'] == seniority].groupby(['ENDDATE']).size().reset_index(name=f'peer_leave_count_{seniority}')
        peer_hires['Brand'] = brand
        peer_leaves['Brand'] = brand

        peer_peer_hires = filtered_peer_peer_df[filtered_peer_peer_df['SENIORITY_LEVEL'] == seniority].groupby(['STARTDATE']).size().reset_index(name=f'peer_peer_hire_count_{seniority}')
        peer_peer_leaves = filtered_peer_peer_df[filtered_peer_peer_df['SENIORITY_LEVEL'] == seniority].groupby(['ENDDATE']).size().reset_index(name=f'peer_peer_leave_count_{seniority}')
        peer_peer_hires['Brand'] = brand
        peer_peer_leaves['Brand'] = brand

        peer_hires_dict[seniority] = peer_hires
        peer_leaves_dict[seniority] = peer_leaves
        peer_peer_hires_dict[seniority] = peer_peer_hires
        peer_peer_leaves_dict[seniority] = peer_peer_leaves

    for seniority in [1, 2, 3, 4, 5]:
        # Sum of hires and leaves by date
        peer_hires_sum = peer_hires_dict[seniority].groupby(['Brand', 'STARTDATE']).agg({f'peer_hire_count_{seniority}': 'sum'}).reset_index()
        peer_leaves_sum = peer_leaves_dict[seniority].groupby(['Brand', 'ENDDATE']).agg({f'peer_leave_count_{seniority}': 'sum'}).reset_index()
        peer_peer_hires_sum = peer_peer_hires_dict[seniority].groupby(['Brand', 'STARTDATE']).agg({f'peer_peer_hire_count_{seniority}': 'sum'}).reset_index()
        peer_peer_leaves_sum = peer_peer_leaves_dict[seniority].groupby(['Brand', 'ENDDATE']).agg({f'peer_peer_leave_count_{seniority}': 'sum'}).reset_index()

        # Number of unique ULTIMATE_PARENT_RCID for peers and peers of peers by date
        peer_hires_sum['num_peers'] = len(peer_df['ULTIMATE_PARENT_RCID'].unique())
        peer_leaves_sum['num_peers'] = len(peer_df['ULTIMATE_PARENT_RCID'].unique())
        peer_peer_hires_sum['num_peers_of_peers'] = len(peer_peer_df['ULTIMATE_PARENT_RCID'].unique())
        peer_peer_leaves_sum['num_peers_of_peers'] = len(peer_peer_df['ULTIMATE_PARENT_RCID'].unique())

        # Average hires and leaves by date
        peer_hires_sum[f'avg_peer_hires_{seniority}'] = peer_hires_sum[f'peer_hire_count_{seniority}'] / peer_hires_sum['num_peers']
        peer_leaves_sum[f'avg_peer_leaves_{seniority}'] = peer_leaves_sum[f'peer_leave_count_{seniority}'] / peer_leaves_sum['num_peers']
        peer_peer_hires_sum[f'avg_peer_peer_hires_{seniority}'] = peer_peer_hires_sum[f'peer_peer_hire_count_{seniority}'] / peer_peer_hires_sum['num_peers_of_peers']
        peer_peer_leaves_sum[f'avg_peer_peer_leaves_{seniority}'] = peer_peer_leaves_sum[f'peer_peer_leave_count_{seniority}'] / peer_peer_leaves_sum['num_peers_of_peers']

        # Append to the aggregated dataframes
        aggregated_peer_hire = pd.concat([aggregated_peer_hire, peer_hires_sum], ignore_index=True)
        aggregated_peer_leave = pd.concat([aggregated_peer_leave, peer_leaves_sum], ignore_index=True)
        aggregated_peer_peer_hire = pd.concat([aggregated_peer_peer_hire, peer_peer_hires_sum], ignore_index=True)
        aggregated_peer_peer_leave = pd.concat([aggregated_peer_peer_leave, peer_peer_leaves_sum], ignore_index=True)

    loop_end_time = time.time()
    print("Time taken for processing brand", brand, ":", loop_end_time - loop_start_time, "seconds")

# Save aggregated data
aggregated_peer_hire.to_csv(os.path.join(directory, 'aggregated_peer_hire.csv'), index=False)
aggregated_peer_peer_hire.to_csv(os.path.join(directory, 'aggregated_peer_peer_hire.csv'), index=False)
aggregated_peer_leave.to_csv(os.path.join(directory, 'aggregated_peer_leave.csv'), index=False)
aggregated_peer_peer_leave.to_csv(os.path.join(directory, 'aggregated_peer_peer_leave.csv'), index=False)
'''