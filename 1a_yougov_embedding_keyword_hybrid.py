import pandas as pd
import os
import re
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math

# Define directories
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

# Load data
df = pd.read_csv(os.path.join(tdirectory, "yougov_mkt_exe_job_history_no_seniority.csv"), lineterminator='\n', low_memory=False)
# Preprocess function
def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip() if isinstance(text, str) else ""

df['JOBTITLE_CLEANED'] = df['JOBTITLE_RAW'].apply(preprocess_text)

# Load precomputed embeddings
embedding_df = pd.read_csv(os.path.join(directory, "unique_job_titles_with_embeddings.csv"))
embedding_df['EMBEDDING'] = embedding_df['EMBEDDING'].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))
#embedding_df_unique = embedding_df[['JOBTITLE_CLEANED', 'EMBEDDING']].drop_duplicates(subset='JOBTITLE_CLEANED')
#embedding_df = embedding_df_unique.copy()

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

digital_keywords = ["digital", "creator", "influencer", "mobile", "platform", "web", "online", "social media", "e-commerce",
                    "analytics", "adobe", "facebook", "google", "instagram", "microsoft", "twitter", "youtube", 
                    "animation", "androids", "ios", "cloud", "cyber", "data", "developer", 
                    "software", "user interface", "graphic", "visualization" 
                    ]

# Optimized batch embeddings function
def get_batch_embeddings(text_list, batch_size=128):
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

digital_embeddings = generate_anchor_embeddings(digital_keywords)

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

embedding_df["digital"] = embedding_df["JOBTITLE_CLEANED"].apply(
    lambda x: 1 if isinstance(x, str) and any(phrase.lower() in x.lower() for phrase in digital_keywords) else 0
)
# 1) Separate out your current digital vs. non-digital sets
digital_df = embedding_df[embedding_df["digital"] == 1].copy()
non_digital_df = embedding_df[embedding_df["digital"] == 0].copy()

# 2) Compute the centroid of the DIGITAL group
digital_embeddings = torch.stack(digital_df['EMBEDDING'].tolist())
digital_centroid = digital_embeddings.mean(dim=0)  # shape: [embedding_dim]

# 3) For each job title in non_digital_df, compute similarity to digital_centroid
non_digital_embeddings = torch.stack(non_digital_df['EMBEDDING'].tolist())  # shape: [N, embedding_dim]
# Use PyTorch cosine_similarity on entire batch
# This will produce a 1D tensor of size N, each entry is the similarity to digital_centroid
similarities = F.cosine_similarity(non_digital_embeddings, digital_centroid.unsqueeze(0), dim=1)
# Convert similarities back to numpy if you want to manipulate in pandas
similarities_np = similarities.cpu().numpy()
non_digital_df['similarity'] = similarities_np

# 4) Sort non-digital job titles by their similarity descending
non_digital_df = non_digital_df.sort_values(by='similarity', ascending=False)

# 5) Reclassify the top 10% as digital
top_10pct_count = int(0.10 * len(non_digital_df))
threshold_similarity = non_digital_df.iloc[top_10pct_count - 1]['similarity']
print(f"Threshold cosine similarity for the top 10%: {threshold_similarity:.4f}")
top_10pct_index = non_digital_df.head(top_10pct_count).index

# Compute the largest distance from the centroid for the digital job titles
digital_distances = 1 - F.cosine_similarity(digital_embeddings, digital_centroid.unsqueeze(0), dim=1).cpu().numpy()
max_digital_distance = digital_distances.max()
print(f"Maximum distance from the digital centroid: {max_digital_distance:.4f}")

# Compute cosine similarities for the DIGITAL group
digital_similarities = F.cosine_similarity(digital_embeddings, digital_centroid.unsqueeze(0), dim=1)

# Plot the distribution of similarity values
plt.figure(figsize=(10, 6))
plt.hist(non_digital_df['similarity'], bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(x=threshold_similarity, color='red', linestyle='--', label=f"Threshold: {threshold_similarity:.4f}")
plt.title("Distribution of Cosine Similarity Values")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()

# Save the plot as an image
plot_output_path = os.path.join(directory, "cosine_similarity_distribution_avg_check.png")
plt.savefig(plot_output_path, format='png', dpi=300)
print(f"Plot saved to {plot_output_path}")

# Classify non-digital job titles based on the average similarity threshold
non_digital_df['is_digital'] = non_digital_df['similarity'] >= threshold_similarity

# Update the digital column in the original DataFrame
embedding_df.loc[non_digital_df['is_digital'].index, 'digital'] = non_digital_df['is_digital'].astype(int)

print("Updated digital classifications based on average similarity threshold.")

embedding_df['digital'] = embedding_df['digital'].astype('Int64')


# Apply rule-based classification
embedding_df['SENIORITY_LEVEL'] = embedding_df['JOBTITLE_CLEANED'].apply(classify_seniority)

# Extract unique job titles with their rule-based seniority level
rule_based_titles_df = embedding_df[['JOBTITLE_CLEANED', 'SENIORITY_LEVEL']].drop_duplicates(subset='JOBTITLE_CLEANED')

# Save to an intermediate file
intermediate_output_file = os.path.join(directory, "unique_job_titles_with_rule_based_seniority.csv")
rule_based_titles_df.to_csv(intermediate_output_file, index=False)
print(f"Intermediate file with rule-based seniority levels saved to {intermediate_output_file}")

# Filter job titles that still need to be clustered
unclassified = embedding_df[embedding_df['SENIORITY_LEVEL'] == -1].copy()
job_title_embeddings = torch.stack([emb for emb in unclassified['EMBEDDING']])

# Define pre-assigned level embeddings
level_embeddings = {
    level: torch.stack(embedding_df[embedding_df['SENIORITY_LEVEL'] == level]['EMBEDDING'].tolist())
    for level in range(1, 6)
}

# Optimized similarity threshold calculation using NumPy broadcasting
def calculate_centroid_similarity(level_1_embeddings, level_5_embeddings):
    # Compute the centroids (mean embedding) for each level
    centroid_level1 = level_1_embeddings.mean(dim=0)
    centroid_level5 = level_5_embeddings.mean(dim=0)
    # Compute cosine similarity between the two centroids
    similarity = F.cosine_similarity(centroid_level1.unsqueeze(0), centroid_level5.unsqueeze(0), dim=1).item()
    return similarity

similarity_threshold = 1.28 * calculate_centroid_similarity(level_embeddings[1], level_embeddings[5])
print(f"Similarity threshold for outlier detection: {similarity_threshold}")

# Calculate level centroids
# Adjust level_embeddings if you only want levels 1-4
level_embeddings = {level: embeddings for level, embeddings in level_embeddings.items() if level in range(1, 5)}
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
digital_mapping = final_df_unique.set_index('JOBTITLE_CLEANED')['digital'].to_dict()

# Map the values onto your main DataFrame
df['SENIORITY_LEVEL'] = df['JOBTITLE_CLEANED'].map(seniority_mapping)
df['digital'] = df['JOBTITLE_CLEANED'].map(digital_mapping)
df['digital'] = df['digital'].astype('Int64')
#df = df.merge(final_df[['JOBTITLE_CLEANED', 'SENIORITY_LEVEL', 'digital']], on='JOBTITLE_CLEANED', how='left')

#df = df[~df['MAPPED_ROLE'].isin(select_roles)]

# Save data and outputs
output_file = os.path.join(directory, "yougov_mkt_exe_job_history_digital.csv")
#output_file = os.path.join(directory, "yougov_mkt_exe_job_history_5l_128a_digitalall_8254_check.csv")
df.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")

reshaped_data = {}
unique_digital_values = df['digital'].dropna().unique()  # Ensure you capture unique digital values (0 or 1)

for seniority in [-1, 1, 2, 3, 4, 5]:
    for digital_value in unique_digital_values:
        # Filter data based on both SENIORITY_LEVEL and digital value
        job_titles = df[
            (df['SENIORITY_LEVEL'] == seniority) & (df['digital'] == digital_value)
        ]['JOBTITLE_RAW'].dropna().unique().tolist()
        
        # Split job titles into columns if needed
        num_columns = math.ceil(len(job_titles) / 1048500)
        job_title_columns = {f'JobTitle_Column_{i+1}': job_titles[i*1048500:(i+1)*1048500] for i in range(num_columns)}
        
        # Ensure all columns are of equal length
        max_length = max(len(column) for column in job_title_columns.values()) if job_title_columns else 0
        job_title_columns = {k: v + [None] * (max_length - len(v)) for k, v in job_title_columns.items()}
        
        # Create a DataFrame for the current SENIORITY_LEVEL and digital combination
        reshaped_data[(seniority, digital_value)] = pd.DataFrame(job_title_columns)

output_file_path = os.path.join(directory, 'complete_job_titles_digital.xlsx')

with pd.ExcelWriter(output_file_path) as writer:
    for (seniority, digital_value), data in reshaped_data.items():
        sheet_name = f'Seniority_{seniority}_Digital_{digital_value}'
        data.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Cluster summary saved to {output_file_path}")

'''
reshaped_data = {}
for seniority in [-1,1,2,3,4,5]:
    job_titles = df[df['SENIORITY_LEVEL'] == seniority]['JOBTITLE_RAW'].dropna().unique().tolist()
    num_columns = math.ceil(len(job_titles) / 1048500)
    job_title_columns = {f'JobTitle_Column_{i+1}': job_titles[i*1048500:(i+1)*1048500] for i in range(num_columns)}
    max_length = max(len(column) for column in job_title_columns.values())
    job_title_columns = {k: v + [None] * (max_length - len(v)) for k, v in job_title_columns.items()}
    reshaped_data[seniority] = pd.DataFrame(job_title_columns)

output_file_path = os.path.join(directory, 'complete_job_titles_5l_128a_digital012.xlsx')
with pd.ExcelWriter(output_file_path) as writer:
    for seniority, data in reshaped_data.items():
        data.to_excel(writer, sheet_name=f'Seniority_{seniority}', index=False)
print(f"Cluster summary saved to {output_file_path}")
'''
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
