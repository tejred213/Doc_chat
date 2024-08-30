import ollama
import chromadb
from typing import List, Dict, Generator

# Initialize ChromaDB client
client = chromadb.HttpClient(host='localhost', port=8001)
session_collection = client.get_or_create_collection(name="sessions")


# Generate chat response with session management
def get_chat_response(question: str, collections: List[str], session_id: str) -> Generator[str, None, None]:
    """
    Generate a chat response based on the provided question, document collections, and session ID.

    Parameters:
    question (str): The question to ask.
    collections (List[str]): The list of collections to query.
    session_id (str): The ID of the current session.

    Yields:
    str: A portion of the chat response.
    """
    try:
        SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
        prompt_embedding = ollama.embeddings(
            model="all-minilm", prompt=question)["embedding"]

        # Collect results from all specified collections
        results_list = []
        for collection_name in collections:
            collection = client.get_collection(name=collection_name)
            results = collection.query(query_embeddings=[prompt_embedding], n_results=5)
            results_list.append(results)

        # Combine results and select the top chunks
        top_chunks = combine_and_select_top_chunks(results_list)

        

        # Generate response based on selected chunks
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n".join(top_chunks),
                },
                {"role": "user", "content": question},
            ],
            stream=True
        )

        for chunk in response:
            # Update session with the assistant's response
            yield chunk["message"]["content"]
    except Exception as e:
        print(f"Error generating chat response: {e}")
        yield ""

# Function to combine results and pick top chunks
def combine_and_select_top_chunks(results_list, top_n=7):
    """
    Combine results from multiple collections and select the top N chunks based on similarity.

    Parameters:
    results_list (List[dict]): A list of results from different collections.
    top_n (int): The number of top chunks to select. Default is 7.

    Returns:
    List[str]: The top N chunks of text.
    """
    try:
        combined_results = []
        for result in results_list:
            distances = result.get("distances", [])[0]
            documents = result.get("documents", [])[0]
            combined_results.extend(zip(distances, documents))
        
        # Sort combined results by distance (similarity score)
        combined_results.sort(key=lambda x: x[0])
        
        # Select top N results
        top_chunks = [doc for _, doc in combined_results[:top_n]]
        return top_chunks
    except Exception as e:
        print(f"Error combining and selecting top chunks: {e}")
        return []

