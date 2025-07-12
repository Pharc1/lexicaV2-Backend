from flask import Blueprint, jsonify, request
import logging
import os
from utils import process_pdf, insert_to_chroma
from storage import save_uploaded_file, get_sources_history, delete_source, save_uploaded_text
import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document

documents = Blueprint('documents', __name__)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration ChromaDB
chroma_host = os.getenv('CHROMA_DB_HOST', 'localhost')  
chroma_port = int(os.getenv('CHROMA_DB_PORT', 8000))

try:
    if chroma_host == 'localhost':
        chroma_client = chromadb.Client()
    else:
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
except Exception as e:
    logger.error(f"Erreur de connexion à ChromaDB : {str(e)}")
    chroma_client = chromadb.Client()

@documents.route("file", methods=["POST"])
def upload_file():
    """Endpoint pour uploader et traiter un fichier PDF."""
    try:
        if 'file' not in request.files:
            logger.warning("Aucun fichier trouvé dans la requête.")
            return jsonify({"error": "Aucun fichier trouvé"}), 400

        file = request.files['file']

        if file.filename == '':
            logger.warning("Aucun fichier sélectionné.")
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        if not file.filename.lower().endswith('.pdf'):
            logger.warning("Le fichier n'est pas un PDF.")
            return jsonify({"error": "Seuls les fichiers PDF sont acceptés"}), 400

        logger.info(f"Nom du fichier : {file.filename}")

        # Traitement du PDF avec LangChain (retourne des Documents LangChain)
        langchain_documents = process_pdf(file)
        
        # Convertir les documents LangChain en format pour save_uploaded_file
        documents_content = []
        for i, doc in enumerate(langchain_documents):
            documents_content.append({
                'content': doc.page_content,
                'chunk_index': i,
                'filename': file.filename,
                'metadata': doc.metadata
            })
        
        # Sauvegarder le fichier source et ses métadonnées
        file_path, metadata_path, metadata_filename = save_uploaded_file(file, documents_content)
        
        # Insertion dans ChromaDB
        insert_to_chroma(langchain_documents, chroma_client)

        return jsonify({
            "message": "Fichier traité avec succès",
            "filename": file.filename,
            "documents_processed": len(langchain_documents),
            "chunks_created": len(langchain_documents),
            "file_saved": file_path is not None,
            "metadata_saved": metadata_path is not None
        }), 200

    except Exception as e:
        logger.error("Erreur lors du traitement du fichier PDF : %s", str(e))
        return jsonify({
            "error": f"Erreur lors du traitement du fichier PDF : {str(e)}"
        }), 500


@documents.route("/text",methods=["POST"])
def upload_text():
    """ Endpoint pour uploader du texte dans la vdb"""
    try:

        data = request.get_json()
        text = data.get("text")

        if text == '':
            logger.warning("Texte est vide.")
            return jsonify({"error": "Aucun texte entré"}), 400


        filename = text[:10]+"..."
        # Traitement du PDF avec LangChain (retourne des Documents LangChain)
        document = [Document(page_content=text, metadata={'filename': filename})]
        
        # Convertir les documents LangChain en format pour save_uploaded_file
        documents_content = []
        for i, doc in enumerate(document):
            documents_content.append({
                'content': doc.page_content,
                'chunk_index': i,
                'filename': filename,
                'metadata': doc.metadata
            })
        
        # Sauvegarder le fichier source et ses métadonnées
        file_path, metadata_path, metadata_filename = save_uploaded_text(text)
        
        # Insertion dans ChromaDB
        insert_to_chroma(document, chroma_client)

        return jsonify({
            "message": "Fichier traité avec succès",
            "filename": filename,
            "documents_processed": len(document),
            "chunks_created": len(document),
            "file_saved": file_path is not None,
            "metadata_saved": metadata_path is not None
        }), 200

    except Exception as e:
        logger.error("Erreur lors du traitement du fichier PDF : %s", str(e))
        return jsonify({
            "error": f"Erreur lors du traitement du fichier PDF : {str(e)}"
        }), 500
    
@documents.route("status", methods=["GET"])
def get_status():
    """Endpoint pour vérifier le statut des documents dans ChromaDB."""
    try:
        embeddings_model = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small", 
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        collection = chroma_client.get_or_create_collection(
            name="Documents", 
            embedding_function=embeddings_model
        )
        
        count = collection.count()
        
        return jsonify({
            "documents_count": count,
            "collection_name": "Documents"
        }), 200

    except Exception as e:
        logger.error("Erreur lors de la vérification du statut : %s", str(e))
        return jsonify({"error": "Erreur lors de la vérification du statut"}), 500

@documents.route("history/sources", methods=["GET"])
def get_sources():
    """Récupère l'historique des fichiers sources."""
    try:
        sources = get_sources_history()
        return jsonify({
            "sources": sources,
            "count": len(sources)
        }), 200
    except Exception as e:
        logger.error("Erreur lors de la récupération de l'historique des sources : %s", str(e))
        return jsonify({"error": "Erreur lors de la récupération de l'historique des sources"}), 500

@documents.route("history/sources/<source_id>", methods=["DELETE"])
def delete_source_endpoint(source_id):
    """Supprime un fichier source."""
    try:
        success = delete_source(source_id)
        if success:
            return jsonify({"message": "Source supprimée avec succès"}), 200
        else:
            return jsonify({"error": "Source non trouvée"}), 404
    except Exception as e:
        logger.error("Erreur lors de la suppression de la source : %s", str(e))
        return jsonify({"error": "Erreur lors de la suppression de la source"}), 500