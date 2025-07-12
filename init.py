import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from utils import load_documents_from_folder
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connexion au client Chroma
client = chromadb.HttpClient(host=os.getenv("CHROMA_DB_HOST"), port=8000)
logging.info("heartbeat %d", client.heartbeat())

# Charger les documents et découper en morceaux
folder_path = 'docs'
documents = load_documents_from_folder(folder_path)

# Création d'une fonction d'embedding
embeddings_model = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small", 
    api_key=os.getenv('OPENAI_API_KEY')
)

# Création ou récupération d'une collection
collection = client.get_or_create_collection(name="Documents", embedding_function=embeddings_model)

# Récupérer tous les IDs existants dans la collection et les supprimer
try:
    # Récupérer les IDs
    existing_docs = collection.get()
    existing_ids = existing_docs['ids']

    # Supprimer les documents par ID si la collection n'est pas vide
    if existing_ids:
        collection.delete(ids=existing_ids)
        logging.info("Tous les documents existants ont été supprimés (%d documents).", len(existing_ids))
    else:
        logging.info("Aucun document à supprimer, la collection est déjà vide.")
except Exception as e:
    logging.error("Erreur lors de la suppression des documents : %s", str(e))

# Ajouter les nouveaux documents avec leurs embeddings
try:
    for i, doc in enumerate(documents):
        # Génération d'un ID unique pour chaque document
        doc_id = f"doc_{i}"
        
        # Ajouter le document à la collection
        collection.add(
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            ids=[doc_id]
        )

    # Persister la base de données
    logging.info("Sauvegardé %d morceaux dans la base de données", len(documents))

except Exception as e:
    logging.error("Erreur lors de l'initialisation de Chroma : %s", str(e))

# Requête de test
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=3 # Nombre de résultats à retourner
)

# Construction du contexte
context = "\n\n----\n\n".join(doc for doc in results['documents'][0])
print("context:", context)
print("results:", results)