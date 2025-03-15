import pandas as pd
import os
import time
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import torch

# Define directories
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

# Load data
df = pd.read_csv(os.path.join(tdirectory, "yougov_mkt_exe_job_history_no_seniority.csv"))

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

# Preprocess function with added type-checking for strings
def preprocess_text(text):
    if isinstance(text, str):  # Ensure it's a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()
    else:
        return ""  # Convert non-string values to an empty string

# Apply preprocessing to the job titles column
df['JOBTITLE_CLEANED'] = df['JOBTITLE_RAW'].apply(preprocess_text)

# Ensure all job titles are strings and extract unique job titles
unique_job_titles_df = df[['JOBTITLE_CLEANED']].drop_duplicates()
unique_job_titles = unique_job_titles_df['JOBTITLE_CLEANED'].dropna().unique()  # Remove NaN values
unique_job_titles = [str(title) for title in unique_job_titles]  # Convert to string
#print(f"Number of unique job titles and salaries: {len(unique_job_salary_df)}")
print(f"Number of unique job titles: {len(unique_job_titles)}")

# Load the tokenizer and model, using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("distilroberta-base").to(device)

# Batch processing function to get embeddings
def get_batch_embeddings(text_list, batch_size=64):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings.append(batch_embeddings.cpu())  
    return torch.cat(embeddings)

# Generate embeddings for unique job titles
start_time = time.time()
unique_embeddings = get_batch_embeddings(unique_job_titles)
end_time = time.time()
print(f"Time taken to generate embeddings for all unique titles: {end_time - start_time:.2f} seconds")

# Save unique job titles and their embeddings in a separate file
embedding_df = pd.DataFrame({
    'JOBTITLE_CLEANED': unique_job_titles,
    'EMBEDDING': [embedding.tolist() for embedding in unique_embeddings]  # Convert tensor to list for CSV
})
unique_job_titles_df = unique_job_titles_df.merge(embedding_df, on='JOBTITLE_CLEANED', how='left')

embedding_file = os.path.join(directory, "unique_job_titles_with_embeddings.csv")
unique_job_titles_df.to_csv(embedding_file, index=False)
print(f"Unique embeddings saved to {embedding_file}")

# Reload embeddings as a dictionary for similarity calculation
#embedding_df['EMBEDDING'] = embedding_df['EMBEDDING'].apply(lambda x: torch.tensor(x))
#job_title_to_embedding = dict(zip(embedding_df['JOBTITLE_CLEANED'], embedding_df['EMBEDDING']))

# Generate embeddings for anchor titles
#anchor_embeddings = {title: get_batch_embeddings([title])[0] for title in anchor_titles}

# Calculate cosine similarity for each job title embedding to each anchor embedding
#def calculate_similarity_matrix(job_embeddings, anchor_embeddings):
#    # Convert anchor embeddings to a single 2D tensor
#    anchor_stack = torch.stack(list(anchor_embeddings.values()))
#    similarity_matrix = []
#    for job_emb in job_embeddings:
#        similarity_scores = cosine_similarity(job_emb.view(1, -1), anchor_stack.numpy())
#        similarity_matrix.append(similarity_scores[0])  # Flatten the result for appending
#    return pd.DataFrame(similarity_matrix, columns=anchor_titles)

# Generate similarity matrix for unique job titles
#job_embeddings = list(job_title_to_embedding.values())
#similarity_matrix = calculate_similarity_matrix(job_embeddings, anchor_embeddings)

# Perform clustering on the similarity matrix
#clustering_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
#clusters = clustering_model.fit_predict(similarity_matrix)

# Assign clusters to unique job titles and save separately
#unique_job_titles_df = pd.DataFrame({
#    'JOBTITLE_CLEANED': unique_job_titles,
#    'CLUSTER': clusters
#})
#df = df.merge(unique_job_titles_df, on='JOBTITLE_CLEANED', how='left')

# Save the main dataframe with clusters but without embeddings
#output_file = os.path.join(directory, "yougov_mkt_exe_job_history_with_clusters.csv")
#df.to_csv(output_file, index=False)
#print(f"Clustered data saved to {output_file}")

# Save the cluster summary
#cluster_summary = df.groupby('CLUSTER')['JOBTITLE_CLEANED'].value_counts().groupby(level=0).head(10)
#summary_file = os.path.join(directory, "job_titles_with_clusters_summary.csv")
#cluster_summary.to_csv(summary_file)
#print(f"Cluster summary saved to {summary_file}")
