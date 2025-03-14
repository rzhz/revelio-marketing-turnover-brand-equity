import pandas as pd
import os
from fuzzywuzzy import fuzz, process
import time
import multiprocessing as mp

directory = '/work/SafeGraph/revelio/code'
sdirectory = '/work/SafeGraph/revelio/data'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

n_job_history = 769

j = 2

# Function for fuzzy matching
def fuzzy_match(brand, company_list, scorer=fuzz.WRatio, cutoff=80):
    match = process.extractOne(brand, company_list, scorer=scorer, score_cutoff=cutoff)
    return match if match else (None, 0)

# Function to perform fuzzy matching in parallel
def parallel_fuzzy_match(brand_list, company_list, scorer=fuzz.WRatio, cutoff=80):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(fuzzy_match, [(brand, company_list, scorer, cutoff) for brand in brand_list])
    return results

# List to hold dataframes
dfs = []

# Read and process dataframes
for i in range(32):
    file_path = os.path.join(sdirectory, f"company_ref_00{str(i).zfill(2)}_part_00.parquet")
    df = pd.read_parquet(file_path)
    df['company_nospace'] = df['company'].str.lower().str.replace(' ', '')
    #df['primary_name_nospace'] = df['primary_name'].str.lower().str.replace(' ', '')
    df['child_company_nospace'] = df['child_company'].str.lower().str.replace(' ', '')
    df['ultimate_parent_rcid_name_nospace'] = df['ultimate_parent_rcid_name'].str.lower().str.replace(' ', '')
    dfs.append(df)

# Concatenate all dataframes
private_df = pd.concat(dfs, ignore_index=True)

# Direct Matching
direct_matches = pd.read_csv(os.path.join(directory, "private_brand_rcid_direct_2.csv"))

# Identifying unmatched entries in private_list
unmatched_brands = pd.read_csv(os.path.join(directory, "private_unmatched_brands_2.csv"))
unmatched_brands['Brand'] = unmatched_brands['Brand'].str.lower()
unmatched_brands = unmatched_brands.iloc[(33*j) : (33*(j+1))]

loop_start_time = time.time()
# Identifying unmatched entries in private_df
unmatched_df = private_df[~private_df['rcid'].isin(direct_matches['rcid'])]

# Prepare separate lists for each relevant column
company_list = unmatched_df['company_nospace'].tolist()
#primary_name_list = unmatched_df['primary_name_nospace'].tolist()
child_company_list = unmatched_df['child_company_nospace'].tolist()
ultimate_parent_list = unmatched_df['ultimate_parent_rcid_name_nospace'].tolist()

# Perform fuzzy matching in parallel for each list
brand_list = unmatched_brands['Brand'].tolist()
results_company = parallel_fuzzy_match(brand_list, company_list)
#results_primary_name = parallel_fuzzy_match(brand_list, primary_name_list)
results_child_company = parallel_fuzzy_match(brand_list, child_company_list)
results_ultimate_parent = parallel_fuzzy_match(brand_list, ultimate_parent_list)

# Combine the results and select the best match
unmatched_brands['best_match'] = [
    max(results, key=lambda x: x[1])[0] for results in zip(
        results_company, #results_primary_name, 
        results_child_company, results_ultimate_parent
    )
]

# Filter to include only brands with a best match
fuzzy_matches = unmatched_brands[unmatched_brands['best_match'].notna()]

# Merge fuzzy matches with unmatched_df to get full information
final_fuzzy_matches_company = fuzzy_matches.merge(
    unmatched_df,
    left_on='best_match',
    right_on='company_nospace',
    how='inner'
)

#final_fuzzy_matches_primary_name = fuzzy_matches.merge(
#    unmatched_df,
#    left_on='best_match',
#    right_on='primary_name_nospace',
#    how='inner'
#)

final_fuzzy_matches_child_company = fuzzy_matches.merge(
    unmatched_df,
    left_on='best_match',
    right_on='child_company_nospace',
    how='inner'
)

final_fuzzy_matches_ultimate_parent = fuzzy_matches.merge(
    unmatched_df,
    left_on='best_match',
    right_on='ultimate_parent_rcid_name_nospace',
    how='inner'
)

# Concatenate all the final fuzzy matches
final_fuzzy_matches = pd.concat([
    final_fuzzy_matches_company,
    #final_fuzzy_matches_primary_name,
    final_fuzzy_matches_child_company,
    final_fuzzy_matches_ultimate_parent
]).drop_duplicates()

loop_end_time = time.time()
print("Time taken for one loop iteration:", loop_end_time - loop_start_time, "seconds")
        
# Extract matched company details and add Brand column
#final_fuzzy_matches = unmatched_df.merge(fuzzy_matches, left_on='combined', right_on='fuzzy_matched_company', how='inner')
#final_fuzzy_matches = final_fuzzy_matches[final_fuzzy_matches['Brand'].notna()]

# Save to CSV
final_fuzzy_matches.to_csv(os.path.join(tdirectory, f"private_brand_rcid_fuzzy-{j}.csv"), index=False)
    
        