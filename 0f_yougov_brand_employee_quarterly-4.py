import pandas as pd
import os
import numpy as np

# Define the directory containing the files
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

j = 4

public_list = pd.read_csv(os.path.join(directory, "public_brand_list_revelio_new.csv"))
public_list['conm'] = public_list['conm'].str.lower()

# Placeholder list to store filtered data
job_history = pd.DataFrame()

# Iterate through all the job history CSV files
for i in range(0, 49):
    file_name = os.path.join(tdirectory, f"yougov_public_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False, lineterminator='\n')
        
        merged_df = df.merge(public_list, on='cik', how='inner')
        merged_df = merged_df.rename(columns={'conm': 'firm'})
        job_history = pd.concat([job_history, merged_df], ignore_index=True)       

for i in range(0, 4):
    file_name = os.path.join(tdirectory, f"yougov_private_direct_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
        df = df.rename(columns={'Brand': 'firm'})
        job_history = pd.concat([job_history, df], ignore_index=True)       

for i in range(0, 4):
    file_name = os.path.join(tdirectory, f"yougov_private_fuzzy_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
        df = df.rename(columns={'Brand': 'firm'})
        job_history = pd.concat([job_history, df], ignore_index=True)       

job_history['firm'] = job_history['firm'].str.lower()

# Sort by USER_ID and STARTDATE
job_history = job_history.sort_values(by=['USER_ID', 'STARTDATE'])

def calculate_tenure(row, period_end):
    if pd.notna(row['STARTDATE']) and pd.notna(row['ENDDATE']):
        end_date = row['ENDDATE'] if pd.notna(row['ENDDATE']) else period_end
        return (end_date - row['STARTDATE']).days / 365.25
    else:
        return np.nan

def calculate_employee_info(df):
    df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
    df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')

    df.loc[df['STARTDATE'].notna(), 'START_YEAR'] = df['STARTDATE'].dt.year.astype(pd.Int64Dtype())
    df.loc[df['ENDDATE'].notna(), 'END_YEAR'] = df['ENDDATE'].dt.year.astype(pd.Int64Dtype())

    # Ensure that SALARY is numeric, coercing any errors to NaN
    df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce')

    firms = df['firm'].unique().tolist()

    # Sort data by USER_ID and STARTDATE for prev and next firm calculations
    df = df.sort_values(by=['USER_ID', 'STARTDATE'])

    employee_info = []

    if df['START_YEAR'].notna().any() and df['END_YEAR'].notna().any():
        for firm in firms:
            for year in range(2007 + 2 * j, 2009 + 2 * j):
                for quarter in range(1, 5):
                    # Define the start and end dates for the quarterly period
                    period_start = pd.Timestamp(year, 3 * quarter - 2, 1)
                    period_end = pd.Timestamp(year, 3 * quarter, pd.Timestamp(year, 3 * quarter, 1).days_in_month)

                    # Filter the DataFrame for the current firm
                    firm_df = df[df['firm'] == firm].copy()

                    # Apply calculate_tenure function
                    firm_df['TENURE'] = firm_df.apply(lambda row: calculate_tenure(row, period_end), axis=1)

                    # Identify current employees
                    current_employees = firm_df[(firm_df['STARTDATE'] <= period_end) & (firm_df['ENDDATE'].isna() | (firm_df['ENDDATE'] > period_start))]

                    # Calculate average tenure
                    avg_tenure = current_employees['TENURE'].mean()

                    # Calculate average last period salary for each category, ignoring NaN values
                    avg_salary_current = current_employees['SALARY'].dropna().mean()

                    employee_info.append({
                        'firm': firm,
                        'year': year,
                        'quarter': quarter,
                        'avg_tenure': avg_tenure,
                        'current_employees': len(current_employees),
                        'avg_salary_current': avg_salary_current
                    })

    return pd.DataFrame(employee_info)

# Call the function to calculate employee information
employee_info_df = calculate_employee_info(job_history)
# Save to CSV
employee_info_df.to_csv(os.path.join(tdirectory, f"brand_employee_overall_quarterly_{j}.csv"), index=False)
