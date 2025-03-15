import pandas as pd
import os

# Define the directory containing the files
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

public_list = pd.read_csv(os.path.join(directory, "public_brand_list_revelio_new.csv"))
public_list['conm'] = public_list['conm'].str.lower()

# List of specified "MAPPED_ROLE" values
specified_roles = [
    "sales representative", "sales", "journalist", "graphic designer", "retail salesperson", "customer service representative",
    "writer", "customer service", "marketing", "social media", "artist", "communication", "video editor", "editor", 
    "marketing coordinator", "commercial", "sales consultant", "photographer", "actor", "ambassador", "producer", "designer", 
    "sales account", "graphic design", "seller", "creative", "sales marketing", "sales business development", "key account sales", 
    "marketing consultant", "stage", "translator", "public relations", "salesman", "sales coordinator", "digital marketing", 
    "product", "content creator", "event", "retail sales", "web designer", "animator", "community", "marketing communications",
    "reporter", "service", "content", "promoter", "art", "event coordinator", "communications", "community relations", "sale",
    "sales officer", "marketing officer", "sales agent", "brand ambassador", "campus ambassador", "promotions", "salesperson",
    "design", "brand", "events", "copywriter", "customer service officer", "social media marketing", "marketing communication",
    "marketing sales", "marketing analyst", "advertising", "sales marketing coordinator", "photo", "customer service sales",
    "merchandiser", "communications coordinator", "marketing representative", "business sales", "sales administrator",
    "comercial", "telemarketing", "model", "marketing business development", "sales customer service", "innovation", "digital",
    "market development", "sales promoter", "outside sales representative", "relations", "sales sales", "game designer",
    "commercial sales", "customer", "public affairs", "instructional designer", "customer service agent", "merchandising",
    "media", "pr", "affairs", "retail", "customer service consultant", "hospitality", "communications consultant",
    "graphic artist", "makeup artist", "digital media", "brand marketing", "customer support", "product marketing", "editorial",
    "sales administration", "product designer", "press", "telesales", "telemarketer", "trade marketing",
    "customer service sales representative", "customer service coordinator", "product engineer", "technical writer",
    "product development", "commercial officer", "visual designer", "media relations", "branch sales", "market",
    "brand representative", "design consultant", "outside sales", "product sales", "client service representative",
    "customer success", "visual merchandiser", "marketing research", "merchandise", "costumer service", "ux designer",
    "retail store", "store sales", "marketing project", "customer care", "digital account", "account coordinator",
    "retail sales consultant", "seo", "sales analyst", "digital project", "industrial designer", "market research",
    "sales development representative", "market sales", "sales operations", "strategic marketing", "online marketing",
    "virtual", "campaign", "customer service analyst", "ui ux designer", "contributor", "digital product", "creative services",
    "production artist", "product development engineer", "e commerce", "customer relations", "customer care representative",
    "customer services", "ui designer", "retention", "government affairs", "market research analyst", "sales support",
    "ux ui designer", "interaction designer", "sales development", "customer services agent", "user researcher", "media planner",
    "customer service engineer", "consumer marketing", "digital designer", "promotor", "service sales", "content analyst",
    "sales service", "knowledge", "user designer", "customer relationship", "ecommerce", "product consultant",
    "customer development", "product analyst", "visual merchandising", "sales support representative", "online", "market analyst",
    "customer marketing", "marketing product", "digital sales", "customer consultant", "technical marketing", "merchant",
    "pricing analyst", "crm", "programming", "product design", "sales service representative", "marketing operations",
    "customer support engineer", "customer sales", "customer account", "services sales", "market access",
    "retail sales representative", "products", "media buyer", "retail marketing", "product engineering",
    "customer support representative", "sales operations analyst", "product developer", "pricing", "product support",
    "merchandise planner", "marketing services", "customer representative", "customer engagement", "customer engineer",
    "commercial account", "customer advocate", "customer operations", "retail operations", "account services", "visual", "user",
    "customer solutions", "retail consultant", "retail account"
]

# Placeholder list to store filtered data
job_history_role = pd.DataFrame()
#job_history_category = pd.DataFrame()

# Iterate through all the job history CSV files
for i in range(0,49):
    file_name = os.path.join(tdirectory, f"yougov_public_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False, lineterminator='\n')
        # Filter SENIORITY to only include values 1 through 7
        df['SENIORITY'] = pd.to_numeric(df['SENIORITY'], errors='coerce')
        #df = df[df['SENIORITY'].isin([1, 2, 3, 4, 5, 6, 7])]

        merged_df = df.merge(public_list, on='cik', how='inner')
        merged_df = merged_df[merged_df['conm'].notna()]
        merged_df = merged_df.rename(columns={'conm': 'firm'})
        
        # Filter the data according to MAPPED_ROLE
        filtered_data_role = merged_df[merged_df['MAPPED_ROLE'].isin(specified_roles)]
        #filtered_data_role = filtered_data_role[filtered_data_role['SENIORITY'] > 3]

        # Filter the data according to JOB_CATEGORY
        #filtered_data_category = merged_df[(merged_df['JOB_CATEGORY'] == "Marketing") | (merged_df['JOB_CATEGORY'] == "Sales")]
        #filtered_data_category = filtered_data_category[filtered_data_category['SENIORITY'] > 3]

        job_history_role = pd.concat([job_history_role, filtered_data_role], ignore_index=True)       
        #job_history_category = pd.concat([job_history_category, filtered_data_category], ignore_index=True)       

for i in range(0,4):
    file_name = os.path.join(tdirectory, f"yougov_private_direct_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
        df = df.rename(columns={'Brand': 'firm'})
        #df = df.drop(columns=['Brand'])
        #df = df.rename(columns={'company': 'Brand'})
        # Filter SENIORITY to only include values 1 through 7
        df['SENIORITY'] = pd.to_numeric(df['SENIORITY'], errors='coerce')
        #df = df[df['SENIORITY'].isin([1, 2, 3, 4, 5, 6, 7])]

        # Filter the data according to MAPPED_ROLE
        filtered_data_role = df[df['MAPPED_ROLE'].isin(specified_roles)]
        #filtered_data_role = filtered_data_role[filtered_data_role['SENIORITY'] > 3]

        # Filter the data according to JOB_CATEGORY
        #filtered_data_category = df[(df['JOB_CATEGORY'] == "Marketing") | (df['JOB_CATEGORY'] == "Sales")]
        #filtered_data_category = filtered_data_category[filtered_data_category['SENIORITY'] > 3]

        job_history_role = pd.concat([job_history_role, filtered_data_role], ignore_index=True)       
        #job_history_category = pd.concat([job_history_category, filtered_data_category], ignore_index=True)

for i in range(0,4):
    file_name = os.path.join(tdirectory, f"yougov_private_fuzzy_job_history-{i}.csv")
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
        df = df.rename(columns={'Brand': 'firm'})
        #df = df.drop(columns=['Brand'])
        #df = df.rename(columns={'company': 'Brand'})
        # Filter SENIORITY to only include values 1 through 7
        df['SENIORITY'] = pd.to_numeric(df['SENIORITY'], errors='coerce')
        #df = df[df['SENIORITY'].isin([1, 2, 3, 4, 5, 6, 7])]

        # Filter the data according to MAPPED_ROLE
        filtered_data_role = df[df['MAPPED_ROLE'].isin(specified_roles)]
        #filtered_data_role = filtered_data_role[filtered_data_role['SENIORITY'] > 3]

        # Filter the data according to JOB_CATEGORY
        #filtered_data_category = df[(df['JOB_CATEGORY'] == "Marketing") | (df['JOB_CATEGORY'] == "Sales")]
        #filtered_data_category = filtered_data_category[filtered_data_category['SENIORITY'] > 3]

        job_history_role = pd.concat([job_history_role, filtered_data_role], ignore_index=True)       
        #job_history_category = pd.concat([job_history_category, filtered_data_category], ignore_index=True)

job_history_role = pd.concat([job_history_role, filtered_data_role], ignore_index=True)       
#job_history_category = pd.concat([job_history_category, filtered_data_category], ignore_index=True)

job_history_role.to_csv(os.path.join(tdirectory, "yougov_mkt_exe_job_history_no_seniority.csv"), index=False)
#job_history_category.to_csv(os.path.join(sdirectory, "yougov_mkt_exe_job_history_category.csv"), index=False)