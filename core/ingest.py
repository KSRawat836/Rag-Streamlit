import pypdf
import docx
from config import CHUNK_SIZE, CHUNK_OVERLAP

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

if __name__ == "__main__":
    with open("doc.docx", "rb") as f:
        text = load_file(f)

    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")
    for i in range(len(chunks)):
	    print(f"{i} chunk: {chunks[i]}")
	    print("\n")