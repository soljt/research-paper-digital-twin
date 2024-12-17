from sentence_transformers import SentenceTransformer
import pickle
import vanilla_kb.passages as passages

# Load the saved knowledge base
with open('vanilla_kb/kb_index_embd.pkl', 'rb') as f:
    kb_index_embd = pickle.load(f)

# Initialize the sentence transformer model for querying
model_embd = SentenceTransformer("bert-base-nli-mean-tokens").to("cuda:0")

def get_embd_passages(question, metric='cos', top_k=5):
    query_vector = model_embd.encode(question).squeeze()
    top_indices = kb_index_embd.retrieve(query_vector, metric, k=top_k)
    return [passages.passages[i]['content'] for i in top_indices]

# Example usage
if __name__ == "__main__":
    question = "What are the challenges of digital democracy?"
    results = get_embd_passages(question, metric='cos', top_k=3)
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}\n")