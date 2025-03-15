import pandas as pd
import os
import time
import json
import re

# Define the directory
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'
fdirectory = '/work/SafeGraph/revelio/data/2023latest'


n_job_history = 769

# Load peers_dict from a JSON file
with open(os.path.join(directory, "peers_dict.json"), "r") as json_file:
    peers_dict = json.load(json_file)

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

# Collect all unique RCIDs
all_rcids = set()
for peers_info in peers_dict.values():
    all_rcids.update(peers_info['peers'])
    all_rcids.update(peers_info['peers_of_peers'])
    

# Filter all job history data using the collected RCIDs
filtered_job_history = pd.DataFrame()

for i in range(n_job_history):
    file_name = os.path.join(fdirectory, f"Individual_Job_Positions_and_Job_History-{i}.csv.gz")
    if os.path.exists(file_name):
        job_history_df = pd.read_csv(file_name, low_memory=False)
        df = job_history_df[['USER_ID', 'POSITION_ID', 'COUNTRY', 'STARTDATE', 'ENDDATE', 'JOBTITLE_RAW','MAPPED_ROLE', 'JOB_CATEGORY', 'SENIORITY','RCID', 'ULTIMATE_PARENT_RCID']]
        filtered_df = df[df['COUNTRY'] == 'United States']
        #df['SENIORITY'] = pd.to_numeric(df['SENIORITY'], errors='coerce')
        #df = df[df['SENIORITY'].isin([1, 2, 3, 4, 5, 6, 7])]
        #filtered_df = df[df['SENIORITY'] > 3]

        filtered_df = df[df['RCID'].isin(all_rcids)]
        filtered_df = filtered_df[filtered_df['MAPPED_ROLE'].isin(specified_roles)]

        filtered_job_history = pd.concat([filtered_job_history, filtered_df], ignore_index=True)

filtered_job_history.to_csv(os.path.join(tdirectory, "filtered_job_history_all_peers.csv"), index=False)