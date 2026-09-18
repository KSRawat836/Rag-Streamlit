import pypdf
import docx
from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTORDB_PATH, COLLECTION_NAME
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
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


def upsert_to_vectordb(embed: list[list[float]],chunks: list[str]):
    ids=[]
    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")

    chro = chromadb.PersistentClient(path = VECTORDB_PATH)
    collection=chro.get_or_create_collection(name= COLLECTION_NAME)
    collection.add(ids= ids  , embeddings = embed, documents = chunks)


if __name__ == "__main__":
    with open("doc.docx", "rb") as f:
        text = load_file(f)

    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")
    for i in range(len(chunks)):
        print(f"{i} chunk: {chunks[i]}")
        print("\n")

    embeddings = embed_chunks(chunks)
    embed_list = embeddings.tolist()
    print(type(embeddings))
    print(embeddings.shape)
    print(embeddings)
    upsert_to_vectordb(embed_list, chunks)