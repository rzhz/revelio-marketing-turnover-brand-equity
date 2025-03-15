import pandas as pd
import os
import time
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import torch

# Set directories
directory = '/work/SafeGraph/revelio/code'
tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

i = 5

# Load data
df = pd.read_csv(os.path.join(tdirectory, "filtered_job_history_all_peers.csv"))

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
#print(f"Number of unique job titles and salaries: {len(unique_job_titles_df)}")
print(f"Number of unique job titles: {len(unique_job_titles)}")

# Load the tokenizer and model, using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("distilroberta-base").to(device)

num_parts = 10
titles_per_part = len(unique_job_titles) // num_parts

split_titles = unique_job_titles[i*titles_per_part:(i+1)*titles_per_part]
# Filter `unique_job_salary_df` to only include job titles in `split_titles`
filtered_job_titles_df = unique_job_titles_df[unique_job_titles_df['JOBTITLE_CLEANED'].isin(split_titles)]

#print(f"Number of unique job titles and salaries: {len(unique_job_titles_df)}")
print(f"Number of unique job titles: {len(unique_job_titles)}")

# Batch processing function to get embeddings
def get_batch_embeddings(text_list, batch_size=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings.append(batch_embeddings.cpu())  # Move to CPU to save memory
    return torch.cat(embeddings)

# Generate embeddings
start_time = time.time()
unique_embeddings = get_batch_embeddings(split_titles)
end_time = time.time()
print(f"Time taken to generate embeddings: {end_time - start_time:.2f} seconds")

# Save unique job titles and their embeddings in a separate file
embedding_df = pd.DataFrame({
    'JOBTITLE_CLEANED': split_titles,
    'EMBEDDING': [embedding.tolist() for embedding in unique_embeddings]  # Convert tensor to list for CSV
})
filtered_job_titles_df = filtered_job_titles_df.merge(embedding_df, on='JOBTITLE_CLEANED', how='left')

# Save the results
embedding_file = os.path.join(tdirectory, f"unique_job_titles_with_embeddings_peer-{i}.csv")
filtered_job_titles_df.to_csv(embedding_file, index=False)

print(f"Unique embeddings saved to {embedding_file}")
