from sentence_transformers import SentenceTransformer

def load_embedder():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")