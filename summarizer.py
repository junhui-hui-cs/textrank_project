import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import nltk
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Remove stopwords and tokenize
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    return sentences, words

# Create similarity matrix
def sentence_similarity_matrix(sentences, threshold=0.1):
    # Vectorize sentences using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)
    
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Apply similarity threshold: zero out values below the threshold
    sim_matrix[sim_matrix < threshold] = 0
    
    return sim_matrix

def textrank_with_params(text, top_n=3, damping_factor=0.85, max_iter=100, similarity_threshold=0.1):
    sentences, words = preprocess_text(text)
    
    # Create similarity matrix with threshold
    sim_matrix = sentence_similarity_matrix(sentences, threshold=similarity_threshold)
    
    # Build graph
    graph = nx.from_numpy_array(sim_matrix)
    
    # Apply PageRank (TextRank) with damping factor and max iterations
    scores = nx.pagerank(graph, alpha=damping_factor, max_iter=max_iter)
    
    # Sort sentences by rank scores
    ranked_sentences = sorted(((score, sentence) for sentence, score in zip(sentences, scores.values())), reverse=True)
    
    # Return top n ranked sentences
    return [sentence for score, sentence in ranked_sentences[:top_n]]

# Evaluation
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_summary(original_text, generated_summary, reference_summary=None):
    results = {'generated_summary': generated_summary}

    if reference_summary:
        scores = rouge.score(reference_summary, generated_summary)
        results['rouge1_f1'] = scores['rouge1'].fmeasure
        results['rouge2_f1'] = scores['rouge2'].fmeasure
        results['rougeL_f1'] = scores['rougeL'].fmeasure
    else:
        emb_full = semantic_model.encode(original_text, convert_to_tensor=True)
        emb_sum = semantic_model.encode(generated_summary, convert_to_tensor=True)
        results['semantic_similarity'] = util.pytorch_cos_sim(emb_full, emb_sum).item()

    results['compression_ratio'] = len(generated_summary.split()) / len(original_text.split())
    return results
