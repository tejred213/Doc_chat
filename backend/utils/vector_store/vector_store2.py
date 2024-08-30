import sqlite3

def init_db():
    """Initialize the SQLite database and create necessary tables."""
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()

    # Create tables for embeddings
    c.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_name TEXT NOT NULL,
        doc_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        paragraph TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

init_db()

import pickle  # For serializing embeddings

def sqlite_vector_store(embeddings, paragraphs, collection_name):
    """
    Stores embeddings in an SQLite database.

    Args:
        embeddings (List): List of embeddings to store.
        paragraphs (List[str]): List of text paragraphs corresponding to the embeddings.
        collection_name (str): Name of the collection in the SQLite database.
    """
    try:
        conn = sqlite3.connect('embeddings.db')
        c = conn.cursor()

        # Add embeddings to the database
        for doc_id, (embedding, paragraph) in enumerate(zip(embeddings, paragraphs)):
            # Serialize embedding
            serialized_embedding = pickle.dumps(embedding)
            c.execute('''
                INSERT INTO embeddings (collection_name, doc_id, embedding, paragraph)
                VALUES (?, ?, ?, ?)
            ''', (collection_name, doc_id, serialized_embedding, paragraph))

        conn.commit()
        conn.close()
        print("Stored embeddings in SQLite database")
    except Exception as e:
        print(f"Error storing embeddings in SQLite: {e}")

def delete_from_sqlite(collection_name):
    """
    Delete a collection and its associated embeddings from the SQLite database.

    Parameters:
    collection_name (str): The name of the collection to delete.
    """
    try:
        conn = sqlite3.connect('embeddings.db')
        c = conn.cursor()
        
        # Delete the embeddings for the collection
        c.execute('''
            DELETE FROM embeddings WHERE collection_name = ?
        ''', (collection_name,))
        
        conn.commit()
        conn.close()
        print(f"Collection {collection_name} deleted from SQLite database")
    except Exception as e:
        print(f"Error deleting collection from SQLite: {e}")
