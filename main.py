from flask import Blueprint, jsonify, request, Response
import logging
from openai import OpenAI
from chromadb.utils import embedding_functions
import os
import chromadb
from storage import save_discussion, get_discussions_history, delete_discussion,append_message_to_discussion
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création d'un blueprint Flask pour le module principal
main = Blueprint('main', __name__)

# Initialisation du client OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Définition du chemin pour la base de données Chroma
# Configuration du client Chroma
chroma_host = os.getenv('CHROMA_DB_HOST', 'localhost')  
chroma_port = os.getenv('CHROMA_DB_PORT', 8000) 
chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
embeddings_model = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'))


@main.route('/health')
def health_check():
    """Endpoint de vérification de la santé de l'API."""
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@main.route('/ask', methods=['POST'])
def ask():
    """Traite les demandes de questions de l'utilisateur.

    Returns:
        Response: Réponse générée par l'API OpenAI ou un message d'erreur.
    """
    try:
        # Récupérer la question depuis le body JSON
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Aucune question fournie."}), 400

        question = data.get('question')
        if not question.strip():
            return jsonify({"error": "La question ne peut pas être vide."}), 400

        discussion_path = data.get('filename')
        logger.info(f"Question reçue: {question}")

        base_instructions = """
        Tu es Lexica, un assistant bienveillant qui vouvoie toujours et répond avec joie. 
        Tu fournis uniquement des réponses basées sur tes connaissances. 
        Si une question dépasse tes connaissances, tu l'indiques gentiment. 
        Tu n'as pas toujours besoin du contexte trouvé, tu parles selon le contexte, si tu as des informations mais qu'on te parle naturellement tu parles naturellement aussi.
        Tu utilises tes connaissances trouvées que si besoin.
        Tu réponds toujours avec du markdown tres stylisé et organisé avec toutes sortes de balises afin de rendre le texte agreables des titres des sous tires etc etc emojies aussi si il faut.
        """

        # Recherche dans ChromaDB
        collection = chroma_client.get_or_create_collection(
            name="Documents", 
            embedding_function=embeddings_model
        )

        results = collection.query(
            query_texts=[question],
            n_results=5
        )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        # 💥 Seuil de similarité
        seuil = 1

        # 🔍 Filtrage des résultats pertinents
        filtered_docs = []
        filtered_metas = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            print(dist)
            if dist < seuil:
                filtered_docs.append(doc)
                filtered_metas.append(meta)

        # 💬 Préparation du contexte uniquement si au moins un document pertinent
        if filtered_docs:
            context = "\n\n----\n\n".join(filtered_docs)
            filenames = list({meta.get('filename') for meta in filtered_metas if meta.get('filename')})
            logger.info("Contexte trouvé: %s", context[:200] + "...")
            context_with_instructions = base_instructions + " Connaissances : " + context
        else:
            context = ""
            filenames = []
            logger.info("Aucun contexte assez pertinent trouvé, Lexica répondra sans.")
            context_with_instructions = base_instructions  # pas de contexte injecté

        # Préparer les messages pour OpenAI
        messages = [
            {"role": "system", "content": context_with_instructions},
            {"role": "user", "content": question},
        ]

        # Si discussion déjà en cours
        if data.get("messages"):
            messages = data.get("messages")
            messages.append({"role": "user", "content": question})
            print("Discussion en cours :", messages)
            append_message_to_discussion(discussion_path,{"type": "user", "content": question})
        else:
            # Sauvegarder la discussion des que la question est posé
            save_discussion(question)


        # Variable pour collecter la réponse complète
        full_response = ""
        context_used = results['documents'][0] if results['documents'] else []

        def generate():
            """Génère les réponses de l'API OpenAI en streaming."""
            nonlocal full_response
            try:
                 
                
                
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                

            except Exception as e:
                logger.error("Erreur lors de l'appel du modèle : %s", str(e))
                error_message = "Une erreur est survenue lors du traitement de la question."
                full_response = error_message
                save_discussion(question, error_message, context_used)
                yield error_message
        filename_header = "||".join(filenames)
        # Envoi de la réponse en streaming
        return Response(
            generate(), 
            content_type="text/plain", 
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Access-Control-Allow-Origin": "*",
                "X-Used-Filenames": filename_header
            }
        )

    except Exception as e:
        logger.error("Erreur lors de la recherche de similarité : %s", str(e))
        return jsonify({"error": "Erreur lors de la recherche de similarité."}), 500

@main.route('/history/discussions', methods=['GET'])
def get_discussions():
    """Récupère l'historique des discussions."""
    try:
        limit = request.args.get('limit', 10, type=int)
        discussions = get_discussions_history(limit)
        return jsonify({
            "discussions": discussions,
            "count": len(discussions)
        }), 200
    except Exception as e:
        logger.error("Erreur lors de la récupération de l'historique : %s", str(e))
        return jsonify({"error": "Erreur lors de la récupération de l'historique"}), 500

@main.route('/history/discussions/<discussion_id>', methods=['DELETE'])
def delete_discussion_endpoint(discussion_id):
    """Supprime une discussion."""
    try:
        success = delete_discussion(discussion_id)
        if success:
            return jsonify({"message": "Discussion supprimée avec succès"}), 200
        else:
            return jsonify({"error": "Discussion non trouvée"}), 404
    except Exception as e:
        logger.error("Erreur lors de la suppression de la discussion : %s", str(e))
        return jsonify({"error": "Erreur lors de la suppression de la discussion"}), 500