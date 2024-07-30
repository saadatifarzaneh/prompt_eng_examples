from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Functions
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Using mean of last layer hidden states as the sentence embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_centroid(embeddings):
    return np.mean(embeddings, axis=0)

def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

# Read data from files
bio_sentences = read_file('./data/bio.txt')
cs_sentences = read_file('./data/cs.txt')
test_sentences = read_file('./test.txt')

# Getting embeddings
bio_embeddings = [get_embedding(sentence.strip()) for sentence in bio_sentences]
cs_embeddings = [get_embedding(sentence.strip()) for sentence in cs_sentences]

# Calculating average locations (centroids)
bio_centroid = calculate_centroid(bio_embeddings)
cs_centroid = calculate_centroid(cs_embeddings)

# Classifying new sentences from test file and storing results
classifications = []
for new_sentence in test_sentences:
    new_sentence = new_sentence.strip()
    new_embedding = get_embedding(new_sentence)
    
    # Measuring distances to the centroids
    distance_to_bio = np.linalg.norm(new_embedding - bio_centroid)
    distance_to_cs = np.linalg.norm(new_embedding - cs_centroid)
    
    # Classification based on proximity
    if distance_to_bio < distance_to_cs:
        classification = "Bio"
        color = 'b'
    else:
        classification = "CS"
        color = 'r'
    
    # Print sentence and its classification
    print(f"Sentence: {new_sentence}")
    print(f"Classification: {classification}")
    print(f"Distance to Bio: {distance_to_bio}")
    print(f"Distance to CS: {distance_to_cs}")
    print("")
    
    # Store the classification result
    classifications.append((new_sentence, classification, new_embedding, color))

# Concatenate all embeddings for visualization
all_embeddings = np.concatenate((bio_embeddings, cs_embeddings), axis=0)
all_embeddings = np.squeeze(all_embeddings, axis=1)
print(all_embeddings.shape)
# (number_of_sentences, 768)

# Reduce dimensions
# Set perplexity to a value less than the number of samples
perplexity_value = min(30, len(all_embeddings) - 1)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot the embeddings
# Number of bio and cs sentences
num_bio = len(bio_embeddings)
# Plotting
plt.figure(figsize=(10, 6))

# Plot bio embeddings
plt.scatter(embeddings_2d[:num_bio, 0], embeddings_2d[:num_bio, 1], color='b', label='Bio')

# Plot cs embeddings
plt.scatter(embeddings_2d[num_bio:, 0], embeddings_2d[num_bio:, 1], color='r', label='CS')

plt.title('2D t-SNE of Sentence Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()

# Visualize classification results
# Concatenate embeddings for visualization including test sentences
test_embeddings = np.array([classification[2] for classification in classifications])
all_embeddings_with_test = np.concatenate((bio_embeddings, cs_embeddings, test_embeddings), axis=0)
all_embeddings_with_test = np.squeeze(all_embeddings_with_test, axis=1)

# Apply t-SNE for dimensionality reduction on all embeddings including test sentences
tsne_with_test = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
embeddings_2d_with_test = tsne_with_test.fit_transform(all_embeddings_with_test)

# Plot the embeddings including test sentences
plt.figure(figsize=(10, 6))

# Plot bio embeddings
plt.scatter(embeddings_2d_with_test[:num_bio, 0], embeddings_2d_with_test[:num_bio, 1], color='b', label='Bio')

# Plot cs embeddings
plt.scatter(embeddings_2d_with_test[num_bio:num_bio+len(cs_embeddings), 0], embeddings_2d_with_test[num_bio:num_bio+len(cs_embeddings), 1], color='r', label='CS')

# Plot test sentence embeddings
test_start_index = num_bio + len(cs_embeddings)
for i, classification in enumerate(classifications):
    plt.scatter(embeddings_2d_with_test[test_start_index + i, 0], embeddings_2d_with_test[test_start_index + i, 1], color=classification[3], marker='x', s=100, label='Test Sentence' if i == 0 else "")

plt.title('2D t-SNE of Sentence Embeddings with Test Sentences')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()
