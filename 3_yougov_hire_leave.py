import pandas as pd
import numpy as np
import os

# Define the directory
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

df = pd.read_csv(os.path.join(directory, "yougov_mkt_exe_job_history_digital.csv"), lineterminator='\n', low_memory=False)

# Convert dates to datetime
df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')
df['firm'] = df['firm'].str.lower()

# Combine seniority levels
df['SENIORITY_LEVEL'] = df['SENIORITY_LEVEL'].replace({1: 1, 2: 1, 4: 4, 5: 5})

# Sort by USER_ID and STARTDATE
df = df.sort_values(by=['USER_ID', 'firm', 'STARTDATE'])

# Create a lagged column for the previous firm
df['Previous_firm'] = df.groupby('USER_ID')['firm'].shift(1)

# Filter to keep only rows where the firm has changed or previous firm is missing
filtered_df = df[(df['firm'] != df['Previous_firm']) | df['Previous_firm'].isna()]

filtered_df = filtered_df[filtered_df['SENIORITY_LEVEL'].notna()]
filtered_df['SENIORITY_LEVEL'] = filtered_df['SENIORITY_LEVEL'].astype('Int64')


#filtered_df = filtered_df[filtered_df['SENIORITY'] != 'http://linkedin.com/company/sprint-communications-staffordshire']

print(list(filtered_df.columns.values))
print(filtered_df['SENIORITY_LEVEL'].head())
print(filtered_df['SENIORITY_LEVEL'].unique())
# Get the count of observations for each unique seniority level
seniority_counts = filtered_df['SENIORITY_LEVEL'].value_counts().sort_index()
print("\nNumber of observations for each seniority level:")
print(seniority_counts)
seniority_matrix = pd.crosstab(df['SENIORITY_LEVEL'], df['SENIORITY'])
print("\nMatrix of SENIORITY_LEVEL x SENIORITY:")
print(seniority_matrix)
# Create the matrix using pd.crosstab
seniority_matrix = pd.crosstab(filtered_df['SENIORITY_LEVEL'], filtered_df['SENIORITY'])
print("\nMatrix of SENIORITY_LEVEL x SENIORITY:")
print(seniority_matrix)
filtered_df['SENIORITY_LEVEL'] = pd.to_numeric(filtered_df['SENIORITY_LEVEL']).astype('int64')
print(filtered_df['SENIORITY_LEVEL'].head())

print(df.shape)
print(filtered_df.shape)
print(filtered_df.head())
print(df['firm'].nunique())
print(filtered_df['firm'].nunique())
print(filtered_df['STARTDATE'].head())

# Aggregate for new hires
new_hires = filtered_df.groupby(['firm', 'STARTDATE']).size().reset_index(name='new_hire_count')
multiple_new_hires = new_hires[new_hires['new_hire_count'] > 1]

# Aggregate for leaves
leaves = filtered_df.groupby(['firm', 'ENDDATE']).size().reset_index(name='leave_count')
multiple_leaves = leaves[leaves['leave_count'] > 1]

# Display results
print("Multiple New Hires:\n", multiple_new_hires)
print("\nMultiple Leaves:\n", multiple_leaves)

# Group and aggregate data
aggregated_hire = filtered_df.groupby(['firm', 'cik', 'naics_code', 'STARTDATE', 'SENIORITY_LEVEL']).agg(
    HIRE=('USER_ID', 'count'),
    AVERAGE_SALARY_HIRE=('SALARY', 'mean')
).reset_index()

# Pivot the DataFrame to get SENIORITY levels as separate columns
pivot_hire = aggregated_hire.pivot_table(
    index=['firm', 'cik', 'naics_code', 'STARTDATE'],
    columns='SENIORITY_LEVEL',
    values=['HIRE', 'AVERAGE_SALARY_HIRE'],
    aggfunc='first'  # Using 'first' because after groupby aggregation each cell is already reduced to a single value
)

# Flatten the multi-level columns after pivot
# Ensuring the columns have descriptive names that include the metric and seniority
pivot_hire.columns = ['{}_{}'.format(metric, int(sen)) if isinstance(sen, (float, int)) else '{}'.format(metric)
                      for metric, sen in pivot_hire.columns]
pivot_hire.reset_index(inplace=True)
pivot_hire.fillna(value=0, inplace=True)

# Additional aggregation for totals and overall average
total_hire_agg = filtered_df.groupby(['firm', 'cik', 'naics_code', 'STARTDATE']).agg(
    TOTAL_HIRE=('USER_ID', 'count'),
    OVERALL_AVERAGE_SALARY_HIRE=('SALARY', 'mean')
).reset_index()

# Output the first few rows of each DataFrame to verify the structure
print(pivot_hire.head())
print(total_hire_agg.head())

# Merge the detailed seniority data with the overall totals
merged_hire = pd.merge(pivot_hire, total_hire_agg, on=['firm', 'cik', 'naics_code', 'STARTDATE'], how='outer')

merged_hire.to_csv(os.path.join(directory, "aggregated_job_history_hire_new.csv"), index=False)

# Aggregate for leaves by subgroup
aggregated_leave_subgroups = filtered_df.groupby(['firm', 'cik', 'naics_code', 'ENDDATE', 'SENIORITY_LEVEL']).agg(
    LEAVE=('USER_ID', 'count'),
    AVERAGE_SALARY_LEAVE=('SALARY', 'mean')
).reset_index()

# Pivot the DataFrame to separate metrics by Experience_Group and SENIORITY_LEVEL
pivot_leave_subgroups = aggregated_leave_subgroups.pivot_table(
    index=['firm', 'cik', 'naics_code', 'ENDDATE'],
    columns='SENIORITY_LEVEL',
    values=['LEAVE', 'AVERAGE_SALARY_LEAVE'],
    aggfunc='first'  # Using 'first' because the groupby aggregation already reduces to a single value
)

# Flatten the multi-level columns after pivot
pivot_leave_subgroups.columns = ['{}_{}'.format(metric, int(sen)) if isinstance(sen, (float, int)) else '{}'.format(metric)
                      for metric, sen in pivot_leave_subgroups.columns]
pivot_leave_subgroups.reset_index(inplace=True)
pivot_leave_subgroups.fillna(value=0, inplace=True)

# Additional aggregation for totals and overall averages
total_leave_agg_subgroups = filtered_df.groupby(['firm', 'cik', 'naics_code', 'ENDDATE']).agg(
    TOTAL_LEAVE=('USER_ID', 'count'),
    OVERALL_AVERAGE_SALARY_LEAVE=('SALARY', 'mean')
).reset_index()

# Merge the detailed seniority data with the overall totals
merged_leave = pd.merge(pivot_leave_subgroups, total_leave_agg_subgroups, on=['firm', 'cik', 'naics_code', 'ENDDATE'], how='outer')

# Save the results for leave subgroups
output_file_leave_subgroups = os.path.join(directory, "aggregated_job_history_leave_new.csv")
merged_leave.to_csv(output_file_leave_subgroups, index=False)
print(f"Leave subgroup data saved to {output_file_leave_subgroups}")

'''
# Group and aggregate data
aggregated_leave = filtered_df.groupby(['Brand', 'cik', 'naics_code', 'ENDDATE', 'SENIORITY_LEVEL']).agg(
    LEAVE=('USER_ID', 'count'),
    AVERAGE_SALARY_LEAVE=('SALARY', 'mean'),
    AVERAGE_EXPERIENCE_LEAVE=('Years_Experience', 'mean')
).reset_index()

# Pivot the DataFrame to get SENIORITY levels as separate columns
pivot_leave = aggregated_leave.pivot_table(
    index=['Brand', 'cik', 'naics_code', 'ENDDATE'],
    columns='SENIORITY_LEVEL',
    values=['LEAVE', 'AVERAGE_SALARY_LEAVE', 'AVERAGE_EXPERIENCE_LEAVE'],
    aggfunc='first'  # Using 'first' because after groupby aggregation each cell is already reduced to a single value
)

# Flatten the multi-level columns after pivot
# Ensuring the columns have descriptive names that include the metric and seniority
pivot_leave.columns = ['{}_{}'.format(metric, int(sen)) if isinstance(sen, (float, int)) else '{}'.format(metric)
                      for metric, sen in pivot_leave.columns]
pivot_leave.reset_index(inplace=True)
pivot_leave.fillna(value=0, inplace=True)

# Additional aggregation for totals and overall average
total_leave_agg = filtered_df.groupby(['Brand', 'cik', 'naics_code', 'ENDDATE']).agg(
    TOTAL_LEAVE=('USER_ID', 'count'),
    OVERALL_AVERAGE_SALARY_LEAVE=('SALARY', 'mean'),
    OVERALL_AVERAGE_EXPERIENCE_LEAVE=('Years_Experience', 'mean')
).reset_index()

merged_leave = pd.merge(pivot_leave, total_leave_agg, on=['Brand', 'cik', 'naics_code', 'ENDDATE'], how='outer')
merged_leave.to_csv(os.path.join(directory, "aggregated_job_history_leave_new.csv"), index=False)
'''