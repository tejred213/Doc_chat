import json
import ollama
import chromadb
from typing import List, Generator, Tuple

# In-memory list to store the conversation history
message_history = []

def load_message_history(file_path="message_history.json"):
    """
    Loads the conversation history from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the message history.
    """
    try:
        with open(file_path, 'r') as file:
            global message_history
            message_history = json.load(file)
    except FileNotFoundError:
        message_history = []

def save_message_history(file_path="message_history.json"):
    """
    Saves the conversation history to a JSON file.

    Args:
        file_path (str): The path to the JSON file where the message history will be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(message_history, file, indent=4)

def store_message_history(question: str, response: str, file_name: str):
    """
    Stores the question, response, and file name in the conversation history and saves it.

    Args:
        question (str): The user's question.
        response (str): The generated response.
        file_name (str): The name of the file where the response was found.
    """
    # Check if the question already exists in the message history
    if any(entry["question"] == question and entry["response"] == response for entry in message_history):
        return  # Avoid duplicates
    
    message_history.append({
        "file_name": file_name,
        "question": question,
        "response": response
    })
    save_message_history()

# Load existing message history at the start of the program
load_message_history()

def get_chat_response(question, collections: List[str]) -> Generator[str, None, None]:
    """
    Generate a chat response based on the provided question and document collections.
    Tracks the conversation history.

    Parameters:
    question (str): The question to ask.
    collections (List[str]): The list of collections to query.

    Yields:
    str: A portion of the chat response.
    """
    try:
        SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
        """
        
        # Generate MiniLM embeddings for the question
        prompt_embedding = ollama.embeddings(
            model="all-minilm", prompt=question)["embedding"]

        # Initialize ChromaDB client and collect results from all specified collections
        client = chromadb.HttpClient(host='localhost', port=8001)
        results_list = []
        for collection_name in collections:
            collection = client.get_collection(name=collection_name)
            results = collection.query(query_embeddings=[prompt_embedding], n_results=5)
            results_list.append({"collection_name": collection_name, "results": results})

        # Combine results and select the top 7 chunks
        top_chunks, file_names = combine_and_select_top_chunks(results_list, top_n=7)

        # Generate a response based on the selected chunks
        response_generator = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n".join(top_chunks),
                },
                {"role": "user", "content": question},
            ],
            stream=True
        )

        response = ""
        for chunk in response_generator:
            content = chunk["message"]["content"]
            response += content
            yield content

        # Store the question, response, and associated file name once
        if file_names:
            store_message_history(question, response, file_names[0])

    except Exception as e:
        print(f"Error generating chat response: {e}")
        yield ""

def combine_and_select_top_chunks(results_list, top_n=7) -> Tuple[List[str], List[str]]:
    """
    Combine results from multiple collections and select the top N chunks based on similarity.

    Parameters:
    results_list (List[dict]): A list of results from different collections.
    top_n (int): The number of top chunks to select. Default is 7.

    Returns:
    Tuple[List[str], List[str]]: The top N chunks of text and their corresponding file names.
    """
    try:
        combined_results = []
        file_names = []
        for result in results_list:
            distances = result["results"].get("distances", [])[0]
            documents = result["results"].get("documents", [])[0]
            collection_name = result["collection_name"]
            combined_results.extend(zip(distances, documents, [collection_name]*len(documents)))
        
        # Sort combined results by distance (similarity score)
        combined_results.sort(key=lambda x: x[0])
        
        # Select top N results
        top_chunks = [doc for _, doc, _ in combined_results[:top_n]]
        file_names = [file_name for _, _, file_name in combined_results[:top_n]]
        return top_chunks, file_names
    except Exception as e:
        print(f"Error combining and selecting top chunks: {e}")
        return [], []

# Example usage:
# for response in get_chat_response("What is the capital of France?", ["collection1", "collection2"]):
#     print(response)