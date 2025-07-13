import json
import os
from datetime import datetime
import shutil
from typing import Dict, Any, List

def ensure_directories_exist():
    """Crée les dossiers nécessaires pour le stockage."""
    directories = ['save/discussions', 'save/sources']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_discussion(question: str,response:str, context_used: List[str] = None):
    """
    Sauvegarde une discussion dans un fichier JSON.
    
    Args:
        question: La question posée par l'utilisateur
        response: La réponse générée par l'IA
        context_used: Les documents utilisés comme contexte
    """
    ensure_directories_exist()
    
    timestamp = datetime.now()
    discussion_data = {
        "timestamp": timestamp.isoformat(),
        "question": question,
        "response": response,
        "context_used": context_used or [],
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S")
    }
    
    # Nom du fichier basé sur la date et l'heure
    filename = f"discussion_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('save/discussions', filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(discussion_data, f, ensure_ascii=False, indent=2)
        print(f"Discussion sauvegardée : {filepath}")
        return filename
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la discussion : {str(e)}")
        return None

def append_message_to_discussion(filepath: str, message: dict):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["messages"].append(message)
        else:
            data = {
                "timestamp": datetime.now().isoformat(),
                "messages": [message],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Message ajouté à la discussion: {filepath}")
    except Exception as e:
        print(f"Erreur append message: {e}")
        
def save_uploaded_text(text):
    """
    Sauvegarde un texte brut dans un fichier .txt avec ses métadonnées associées.
    
    Args:
        text (str): Le texte à sauvegarder.
    """
    ensure_directories_exist()
    timestamp = datetime.now()
    
    preview = text[:10].replace('\n', ' ').replace('\r', ' ')
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{preview}...txt"
    file_path = os.path.join("save/sources", filename)

    try:
        # Sauvegarde du texte brut
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Métadonnées
        metadata = {
            "timestamp": timestamp.isoformat(),
            "saved_filename": filename,
            "file_path": file_path,
            "text_length": len(text),
            "preview": preview + "..." if len(text) > 10 else preview,
        }

        metadata_filename = f"metadata_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        metadata_path = os.path.join("save/sources", metadata_filename)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Fichier sauvegardé : {file_path}")
        print(f"Métadonnées : {metadata_path}")

        return file_path, metadata_path, metadata_filename

    except Exception as e:
        print(f" Erreur de sauvegarde : {str(e)}")
        return None, None, None


def save_uploaded_file(file, processed_documents: List[Dict]):
    """
    Sauvegarde le fichier source uploadé et ses métadonnées.
    
    Args:
        file: Le fichier uploadé
        processed_documents: Les documents traités extraits du fichier (avec chunks LangChain)
    """
    ensure_directories_exist()
    
    timestamp = datetime.now()
    
    # Sauvegarde du fichier original
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_path = os.path.join('save/sources', filename)
    
    try:
        # Réinitialiser le pointeur du fichier au début
        file.seek(0)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        
        # Sauvegarde des métadonnées avec informations sur les chunks
        metadata = {
            "timestamp": timestamp.isoformat(),
            "original_filename": file.filename,
            "saved_filename": filename,
            "file_path": file_path,
            "chunks_processed": len(processed_documents),
            "processing_date": timestamp.strftime("%Y-%m-%d"),
            "processing_time": timestamp.strftime("%H:%M:%S"),
            "processing_method": "LangChain chunking",
            "chunk_size": 1024,
            "chunk_overlap": 100,
            "documents_content": [
                {
                    "chunk_index": doc.get('chunk_index', i),
                    "content_length": len(doc['content']),
                    "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc.get('metadata', {})
                }
                for i, doc in enumerate(processed_documents)
            ]
        }
        
        metadata_filename = f"metadata_{timestamp.strftime('%Y%m%d_%H%M%S')}_{file.filename}.json"
        metadata_path = os.path.join('save/sources', metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Fichier source sauvegardé : {file_path}")
        print(f"Métadonnées sauvegardées : {metadata_path}")
        
        return file_path, metadata_path, metadata_filename
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier source : {str(e)}")
        return None, None, None

def get_discussions_history(limit: int = 10) -> List[Dict]:
    """
    Récupère l'historique des discussions.
    
    Args:
        limit: Nombre maximum de discussions à retourner
        
    Returns:
        Liste des discussions récentes avec leurs IDs
    """
    discussions_dir = 'save/discussions'
    if not os.path.exists(discussions_dir):
        return []
    
    discussions = []
    files = sorted(os.listdir(discussions_dir), reverse=True)
    
    for filename in files[:limit]:
        if filename.endswith('.json'):
            filepath = os.path.join(discussions_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    discussion = json.load(f)
                    discussion['id'] = filename.replace('.json', '')
                    discussion['filename'] = filename
                    discussions.append(discussion)
            except Exception as e:
                print(f"Erreur lors de la lecture de {filename} : {str(e)}")
    
    return discussions

def get_sources_history() -> List[Dict]:
    """
    Récupère l'historique des fichiers sources.
    
    Returns:
        Liste des fichiers sources avec leurs métadonnées
    """
    sources_dir = 'save/sources'
    if not os.path.exists(sources_dir):
        return []
    
    sources = []
    files = [f for f in os.listdir(sources_dir) if f.startswith('metadata_') and f.endswith('.json')]
    
    for filename in sorted(files, reverse=True):
        filepath = os.path.join(sources_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                metadata['id'] = filename.replace('.json', '')
                metadata['metadata_filename'] = filename
                sources.append(metadata)
        except Exception as e:
            print(f"Erreur lors de la lecture de {filename} : {str(e)}")
    
    return sources

def delete_discussion(discussion_id: str) -> bool:
    """
    Supprime une discussion.
    
    Args:
        discussion_id: L'ID de la discussion à supprimer
        
    Returns:
        True si la suppression a réussi, False sinon
    """
    discussions_dir = 'save/discussions'
    filename = f"{discussion_id}.json"
    filepath = os.path.join(discussions_dir, filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Discussion supprimée : {filepath}")
            return True
        else:
            print(f"Fichier de discussion non trouvé : {filepath}")
            return False
    except Exception as e:
        print(f"Erreur lors de la suppression de la discussion : {str(e)}")
        return False

def delete_source(source_id: str) -> bool:
    """
    Supprime un fichier source et ses métadonnées.
    
    Args:
        source_id: L'ID du fichier source à supprimer
        
    Returns:
        True si la suppression a réussi, False sinon
    """
    sources_dir = 'save/sources'
    metadata_filename = f"{source_id}.json"
    metadata_filepath = os.path.join(sources_dir, metadata_filename)
    
    try:
        # Lire les métadonnées pour obtenir le nom du fichier source
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Supprimer le fichier source
            source_filepath = metadata.get('file_path', '')
            if source_filepath and os.path.exists(source_filepath):
                os.remove(source_filepath)
                print(f"Fichier source supprimé : {source_filepath}")
            
            # Supprimer les métadonnées
            os.remove(metadata_filepath)
            print(f"Métadonnées supprimées : {metadata_filepath}")
            
            return True
        else:
            print(f"Fichier de métadonnées non trouvé : {metadata_filepath}")
            return False
    except Exception as e:
        print(f"Erreur lors de la suppression de la source : {str(e)}")
        return False
