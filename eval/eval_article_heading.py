from transformers import AutoTokenizer, AutoModel
import torch
import re




def get_article_headings(article):
    """Get headings from the article (all markdown heading levels)"""
    # Extract all markdown headings (##, ###, ####, etc.)
    heading_pattern = r'^#{2,}\s+(.+)$'
    headings = re.findall(heading_pattern, article, re.MULTILINE)
    # Clean up headings (remove extra whitespace)
    headings = [h.strip() for h in headings if h.strip()]
    return headings

#Mean Pooling - take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_headings(headings, tokenizer, model):
    """Embed a list of headings"""
    # Tokenize sentences
    encoded_input = tokenizer(headings, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings



if __name__ == "__main__":
    # soft heading recall
    generated_article = open('../generated_example_DMD.md', 'r').read()
    reference_article = open('../src/article_structure.txt', 'r').read()

    generated_headings = get_article_headings(generated_article)  # Prediction P
    reference_headings = get_article_headings(reference_article)  # Ground truth G

    # Load model from hugging face
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Embed all headings
    generated_embeddings = embed_headings(generated_headings, tokenizer, model)  # P embeddings
    reference_embeddings = embed_headings(reference_headings, tokenizer, model)  # G embeddings

    # Calculate soft count for each reference heading
    # For each Gi in G: soft_count(Gi) = 1 / sum(Sim(Gi, Pj) for all Pj in P)
    # Sim(Gi, Pj) = cos(embed(Gi), embed(Pj))
    soft_counts = []
    for i, ref_emb in enumerate(reference_embeddings):
        # Calculate cosine similarity between ref_emb and all generated embeddings
        similarities = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), generated_embeddings, dim=1
        )
        # Soft count = 1 / sum of similarities
        sum_similarities = torch.sum(similarities)
        soft_count = 1.0 / (sum_similarities + 1e-9)  # Add small epsilon to avoid division by zero
        soft_counts.append(soft_count.item())

    # Soft heading recall: average of soft counts
    soft_heading_recall = sum(soft_counts) / len(soft_counts) if soft_counts else 0.0
    
    print(f"Soft heading recall: {soft_heading_recall}")
    print(f"Number of reference headings: {len(reference_headings)}")
    print(f"Number of generated headings: {len(generated_headings)}")