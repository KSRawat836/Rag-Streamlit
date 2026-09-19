import chromadb
from core.ingest import model
from core.config import VECTORDB_PATH, COLLECTION_NAME, TOP_K

def retrieve(query):
    query_embeddings  = model.encode([query])
    chro = chromadb.PersistentClient(path = VECTORDB_PATH)
    collection = chro.get_or_create_collection(name= COLLECTION_NAME)

    results = collection.query(
        query_embeddings = query_embeddings.tolist(),
        n_results = TOP_K
        )
    return results


if __name__ == "__main__":
    results = retrieve("Who organized the 1956 Dartmouth Conference?")
    print(results['documents'])
    print(results['distances'])

