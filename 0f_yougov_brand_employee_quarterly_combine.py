import pandas as pd
import numpy as np
import os

directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'
num_files = 8

firm_employee_overall = pd.DataFrame()

# Load all data
for i in range(0, num_files):
    df = pd.read_csv(os.path.join(tdirectory, f"brand_employee_overall_quarterly_{i}.csv"), low_memory=False)
    firm_employee_overall = pd.concat([firm_employee_overall, df], ignore_index=True)

# Save the final aggregated data
firm_employee_overall.to_csv(os.path.join(directory, "brand_employee_overall_quarterly.csv"), index=False)
