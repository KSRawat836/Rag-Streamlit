import chromadb
from ingest import model
from config import VECTORDB_PATH, COLLECTION_NAME, TOP_K

def retrieve(query):
    query_embeddings  = model.encode([query])
    chro = chromadb.PersistentClient(path = VECTORDB_PATH)
    collection = chro.get_or_create_collection(name= COLLECTION_NAME)

    results = collection.query(
        query_embeddings = query_embeddings.tolist(),
        n_results = TOP_K
        )
    return results


