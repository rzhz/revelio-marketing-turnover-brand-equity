import pandas as pd
import os
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import math

# Define directories
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

# Load data
job_history_df = pd.read_csv(os.path.join(tdirectory, "filtered_job_history_all_peers.csv"), lineterminator='\n', low_memory=False)
# Preprocess function
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip() if isinstance(text, str) else ""

job_history_df['JOBTITLE_CLEANED'] = job_history_df['JOBTITLE_RAW'].apply(preprocess_text)

# Load precomputed embeddings
num_files = 10
embedding_df = pd.DataFrame()
for i in range(0, num_files):
    df = pd.read_csv(os.path.join(tdirectory, f"unique_job_titles_salary_with_embeddings_peer-{i}.csv"), low_memory=False)
    embedding_df = pd.concat([embedding_df, df], ignore_index=True)
    
embedding_df['EMBEDDING'] = embedding_df['EMBEDDING'].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))

#test
#embedding_df = embedding_df.head(10000)
# Define a function to filter out sales-related roles
def is_excluded_sales_role(job_title):
    """
    Check if the job title contains sales-related keywords and should be excluded.
    """
    if isinstance(job_title, str):  # Ensure job_title is a string
        excluded_keywords = ["sale", "production", "operation", "customer service", "customer care"]
        return any(keyword in job_title.lower() for keyword in excluded_keywords)
    return False

# Remove rows with sales-related job titles
embedding_df = embedding_df[~embedding_df['JOBTITLE_CLEANED'].apply(is_excluded_sales_role)].copy()
df = df[~df['JOBTITLE_CLEANED'].apply(is_excluded_sales_role)].copy()

# Load the tokenizer and model, using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("distilroberta-base").to(device)

# List of anchor titles and acronyms representing the executive level
anchor_titles = [
    "Chief Marketing Officer", "Vice President of Marketing", "Executive Vice President of Marketing",
    "Senior Vice President of Marketing", "Chief Sales Officer", "Chief Revenue Officer",
    "Vice President of Sales", "Executive Vice President of Sales", "Senior Vice President of Sales",
    "Vice President of Revenue", "Executive Vice President of Revenue", "Senior Vice President of Revenue",
    "Chief Business Development Officer", "Vice President of Business Development",
    "Executive Vice President of Business Development", "Senior Vice President of Business Development",
    "Chief Market Development Officer", "Vice President of Market Development",
    "Executive Vice President of Market Development", "Senior Vice President of Market Development",
    "Chief Commercial Officer", "Vice President of Commerce",
    "Executive Vice President of Commerce", "Senior Vice President of Commerce"
]
executive_terms = ["CEO", "CPO", "CMO", "CBO", "CFO", "CTO", "COO", "CDO"]

# Optimized batch embeddings function
def get_batch_embeddings(text_list, batch_size=64):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(outputs.cpu())
    return torch.cat(all_embeddings)

def generate_anchor_embeddings(anchor_titles):
    anchor_text_list = [title.lower() for title in anchor_titles]
    anchor_embeddings = get_batch_embeddings(anchor_text_list)
    return anchor_embeddings

anchor_embeddings = generate_anchor_embeddings(anchor_titles)

# Rule-based classification
# Update the classify_seniority function to use regex with word boundaries
def classify_seniority(job_title):
    if isinstance(job_title, str):  
        job_title_lower = job_title.lower()

        # Compile regex patterns for exact word matching
        cmo_pattern = re.compile(r'(' + '|'.join([re.escape(term.lower()) for term in anchor_titles]) + r')')
        exec_pattern = re.compile(r'\b(' + '|'.join([re.escape(term.lower()) for term in executive_terms]) + r')\b')
        vp_pattern = re.compile(r'\b(evp|svp|vp)|(evp|svp|vp)\b|president|vice president')
        director_pattern = re.compile(r'director')
        manager_pattern = re.compile(r'manager|\bmgr\b')
        assistant_to_pattern = re.compile(r'assistant to')
        junior_level_pattern = re.compile(r'representative|specialist|analyst|coordinator|consultant|artist|design')
        entry_level_pattern = re.compile(r'assistant|support|clerk|intern')
        
        # Executive level - Level 5
        if cmo_pattern.search(job_title_lower) and "assistant" not in job_title_lower and "associate" not in job_title_lower and "clerk" not in job_title_lower and "intern" not in job_title_lower:
            return 5

        # Top management level - Level 4
        if exec_pattern.search(job_title_lower) and "associate" not in job_title_lower and "assistant" not in job_title_lower and "clerk" not in job_title_lower and "intern" not in job_title_lower:
            return 4 
        elif "chief" in job_title_lower and "officer" in job_title_lower and "associate" not in job_title_lower and "assistant" not in job_title_lower and "clerk" not in job_title_lower and "intern" not in job_title_lower:
            return 4  
        elif vp_pattern.search(job_title_lower) and not assistant_to_pattern.search(job_title_lower) and "clerk" not in job_title_lower and "intern" not in job_title_lower:
            return 4
        elif director_pattern.search(job_title_lower) and "assistant" not in job_title_lower and "clerk" not in job_title_lower and "intern" not in job_title_lower:
            return 4

        # Middle management level - Level 3
        if manager_pattern.search(job_title_lower) and not assistant_to_pattern.search(job_title_lower):
            return 3  
        elif director_pattern.search(job_title_lower) and "assistant" in job_title_lower:
            return 3  

        # Junior level - Level 2
        if junior_level_pattern.search(job_title_lower):
            return 2  
        
        # Entry level - Level 1
        if entry_level_pattern.search(job_title_lower):
            return 1
    return -1  

# Apply rule-based classification
embedding_df['SENIORITY_LEVEL'] = embedding_df['JOBTITLE_CLEANED'].apply(classify_seniority)

# Extract unique job titles with their rule-based seniority level
rule_based_titles_df = embedding_df[['JOBTITLE_CLEANED', 'SENIORITY_LEVEL']].drop_duplicates(subset='JOBTITLE_CLEANED')

# Filter job titles that still need to be clustered
unclassified = embedding_df[embedding_df['SENIORITY_LEVEL'] == -1].copy()
job_title_embeddings = torch.stack([emb for emb in unclassified['EMBEDDING']])

# Define pre-assigned level embeddings
level_embeddings = {
    level: torch.stack(embedding_df[embedding_df['SENIORITY_LEVEL'] == level]['EMBEDDING'].tolist())
    for level in range(1, 6)
}


def calculate_centroid_similarity(level_1_embeddings, level_5_embeddings):
    # Compute the centroids (mean embedding) for each level
    centroid_level1 = level_1_embeddings.mean(dim=0)
    centroid_level5 = level_5_embeddings.mean(dim=0)
    # Compute cosine similarity between the two centroids
    similarity = F.cosine_similarity(centroid_level1.unsqueeze(0), centroid_level5.unsqueeze(0), dim=1).item()
    return similarity

similarity_threshold = 0.942650207877159#1.28 * calculate_centroid_similarity(level_embeddings[1], level_embeddings[5])
print(f"Similarity threshold for outlier detection: {similarity_threshold}")

# Calculate level centroids
# Adjust level_embeddings if you only want levels 1-4
level_embeddings = {level: embeddings for level, embeddings in level_embeddings.items() if level in [1,3,4]}
level_centroids = {level: level_embeddings[level].mean(dim=0) for level in level_embeddings}

# Ensure classify_unassigned_with_centroids only affects unclassified titles
unclassified_embeddings = torch.stack(unclassified['EMBEDDING'].tolist())

# Only apply `classify_unassigned_with_centroids` to unclassified titles (seniority level -1)
def classify_unassigned_with_centroids(job_embeddings, level_centroids, threshold):
    classifications = []
    for job_emb in job_embeddings:
        # Calculate similarity to each level centroid
        level_similarities = {
            level: cosine_similarity(job_emb.view(1, -1), centroid.view(1, -1)).item()
            for level, centroid in level_centroids.items()
        }
        
        # Check if all similarities are below threshold, assign level -1 as outlier if true
        if np.all([sim < threshold for sim in level_similarities.values()]):
            classifications.append(-1)
        else:
            # Otherwise, assign the level with highest similarity
            classifications.append(max(level_similarities, key=level_similarities.get))
    return classifications

# Process only unclassified titles
unclassified['SENIORITY_LEVEL'] = classify_unassigned_with_centroids(unclassified_embeddings, level_centroids, similarity_threshold)


final_df = pd.concat([embedding_df[embedding_df['SENIORITY_LEVEL'] != -1], unclassified])

# Ensure that the key column in final_df is unique
final_df_unique = final_df.drop_duplicates(subset='JOBTITLE_CLEANED')

# Create mapping dictionaries
seniority_mapping = final_df_unique.set_index('JOBTITLE_CLEANED')['SENIORITY_LEVEL'].to_dict()
# Map the values onto your main DataFrame
job_history_df['SENIORITY_LEVEL'] = job_history_df['JOBTITLE_CLEANED'].map(seniority_mapping)
#job_history_df = job_history_df.merge(final_df[['JOBTITLE_CLEANED', 'SENIORITY_LEVEL']], on='JOBTITLE_CLEANED', how='left')

# Save data and outputs
output_file = os.path.join(directory, "yougov_mkt_peer_job_history.csv")
job_history_df.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")

reshaped_data = {}
for seniority in [-1,1,2,3,4,5]:
    job_titles = job_history_df[job_history_df['SENIORITY_LEVEL'] == seniority]['JOBTITLE_RAW'].dropna().unique().tolist()
    num_columns = math.ceil(len(job_titles) / 1048500)
    job_title_columns = {f'JobTitle_Column_{i+1}': job_titles[i*1048500:(i+1)*1048500] for i in range(num_columns)}
    max_length = max(len(column) for column in job_title_columns.values())
    job_title_columns = {k: v + [None] * (max_length - len(v)) for k, v in job_title_columns.items()}
    reshaped_data[seniority] = pd.DataFrame(job_title_columns)

output_file_path = os.path.join(directory, 'complete_peer_job_titles.xlsx')
with pd.ExcelWriter(output_file_path) as writer:
    for seniority, data in reshaped_data.items():
        data.to_excel(writer, sheet_name=f'Seniority_{seniority}', index=False)
print(f"Cluster summary saved to {output_file_path}")

# t-SNE visualization
#tsne = TSNE(n_components=2, random_state=0)
#tsne_results = tsne.fit_transform(unclassified_embeddings.numpy())
#final_df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
#final_df_tsne['SENIORITY_LEVEL'] = unclassified['SENIORITY_LEVEL'].values

# Plot clusters and save
#output_directory = os.path.join(directory, "cluster_visualizations_direct_assignment")
#os.makedirs(output_directory, exist_ok=True)

#plt.figure(figsize=(10, 8))
#for level in sorted(final_df_tsne['SENIORITY_LEVEL'].unique()):
#    plt.scatter(
#        final_df_tsne.loc[final_df_tsne['SENIORITY_LEVEL'] == level, 'TSNE1'],
#        final_df_tsne.loc[final_df_tsne['SENIORITY_LEVEL'] == level, 'TSNE2'],
#        label=f'Level {level}', alpha=0.6
#    )
#plt.legend()
#plt.title("Job Title Clusters by Seniority Level (t-SNE Visualization)")
#plt.xlabel("TSNE1")
#plt.ylabel("TSNE2")
#image_path = os.path.join(output_directory, "job_title_clusters_tsne_direct_assignment.png")
#plt.savefig(image_path, format='png', dpi=300)
#print(f"Cluster visualization saved to {image_path}")
#plt.show()
