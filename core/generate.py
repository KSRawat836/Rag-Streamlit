import ollama
from config import OLLAMA_MODEL
from retrieve import retrieve

def generate_answer(results,query):
    context = "\n".join(results['documents'][0])

    prompt = f"""Given the context below, answer the question. If the context does not contain enough information to answer, respond with exactly: NO_CONTEXT
    Context: {context }
    Question: {query}
    """

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role":"user","content":prompt}]
        )
    if "NO_CONTEXT" in response.message.content:
        response2=ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role":"user","content":query}]) 
        return(f"NO CONTEXT FOUND IN THE DOCUMENT \n {response2.message.content}")
    else:
        return response.message.content

if __name__ == "__main__":
    query = "What is the capital of France?"
    results = retrieve(query)
    print(generate_answer(results, query))

