import pandas as pd
import numpy as np
import os

# Paths
directory = '/work/SafeGraph/revelio/code'
sdirectory = '/work/SafeGraph/revelio/data'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

# Read the private list
private = {'Brand': ['bmw', 'mercedes', 'citgo', 'lyft', 'purina', 
                     'espn', 'pfizer', 'applesiri', 'hefty', 'googleassistant',
                     'googlemaps', 'googlefi', 'googlesearch', 'youtubeoriginals',
                     'amazonfresh', 'tinder', 'hinge', 'xfinitymobile', 'disney', 
                     'abott', 'bosch', 'nintendo', 'pigglywiggly', 'adidas',
                     'nestlepurelife', 'airborne'],
           "to_match": ['bmwgroup', 'mercedesbenz', 'citgoholding', 
                        'lyft,inc.', 'nestlé', 'espn,inc.', 'pfizerinc.',
                        'apple,inc.', 'reynoldsconsumerproductsholdingsllc', 
                        'googlellc', 'googlellc', 'googlellc', 'googlellc', 
                        'googlellc', 'amazon.com,inc.', 'matchgroup,inc.', 
                        'matchgroup,inc.', 'comcastcablecommunicationsllc', 
                        'thewaltdisneyco.', 'abbottlaboratories(22141982)',
                        'robertboschgmbh', 'nintendoco.,ltd.', 'pigglywigglyllc',
                        'adidasag', 'nestlé', 'reckittbenckisergroupplc']}
private_list = pd.DataFrame(private)

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

# Create a dictionary to map to_match values to Brand values
brand_dict = dict(zip(private_list['to_match'], private_list['Brand']))

# Direct Matching
direct_matches = cik_df[cik_df['company_nospace'].isin(private_list['to_match']) | 
                        cik_df['primary_name_nospace'].isin(private_list['to_match'])| 
                        cik_df['child_company_nospace'].isin(private_list['to_match'])| 
                        cik_df['ultimate_parent_rcid_name_nospace'].isin(private_list['to_match'])]

# Define a function to find the matching brand
def find_matching_brand(row, brand_dict):
    for column in ['company_nospace', 'primary_name_nospace', 'child_company_nospace', 'ultimate_parent_rcid_name_nospace']:
        if row[column] in brand_dict:
            return brand_dict[row[column]]
    return np.nan

# Add a 'Brand' column to direct_matches
direct_matches['Brand'] = direct_matches.apply(find_matching_brand, axis=1, brand_dict=brand_dict)

# Save the results to a CSV file
direct_matches.to_csv(os.path.join(tdirectory, "private_brand_rcid_fuzzy_manual-11.csv"), index=False)
