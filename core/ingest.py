import pypdf
import docx
from core.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTORDB_PATH, COLLECTION_NAME
from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL
import chromadb 

# model for embedding the chunks
model = SentenceTransformer(EMBEDDING_MODEL)


# Text loader for pdf docs and text files
def load_file(file) -> str:
    filename = file.name
    extension = filename.split(".")[-1].lower()

    if extension == "txt":
        raw_bytes = file.read()
        text = raw_bytes.decode("utf-8")

    elif extension == "pdf":
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    elif extension == "docx":
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return text


# chunking the text retrieved from file, edit the chunk size from config.py
def chunk_text(raw_text) -> list[str]:
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(raw_text):
        end = start + CHUNK_SIZE
        chunk = raw_text[start:end]
        chunks.append(chunk)
        start += step

    return chunks


def embed_chunks(chunks: list[str]):
    embeddings = model.encode(chunks)
    return embeddings

def clear_collection():
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

def upsert_to_vectordb(embeddings, chunks, clear_first=True):
    if clear_first:
        clear_collection()

    chro = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chro.get_or_create_collection(name=COLLECTION_NAME)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)


