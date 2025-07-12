import PyPDF2
import logging
from chromadb.utils import embedding_functions
import os
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)

def chunk_text(documents, chunk_size, chunk_overlap):
    """Découpe les documents en morceaux de texte selon les paramètres spécifiés.

    Args:
        documents (list): Liste des documents à découper.
        chunk_size (int): Taille maximale de chaque morceau.
        chunk_overlap (int): Nombre de caractères à chevaucher entre les morceaux.

    Returns:
        list: Liste des morceaux de texte découpés.
    """
    # Crée le text splitter avec les paramètres spécifiés
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info("Découpé %d documents en %d morceaux", len(documents), len(chunks))
    return chunks

def load_documents_from_folder(folder_path, chunk_size=512):
    """Charge les fichiers .txt depuis un dossier et les découpe en morceaux.

    Args:
        folder_path (str): Chemin du dossier contenant les fichiers texte.
        chunk_size (int): Taille des morceaux de texte.

    Returns:
        list: Liste des morceaux de texte découpés.
    """
    try:
        loader = DirectoryLoader(folder_path, glob="*.txt")
        documents = loader.load()
        # Découpe les documents en morceaux
        documents = chunk_text(documents, 1024, 100)

        # Log l'information sur le dixième document pour avoir un aperçu 
        if len(documents) > 1:  # Vérifie qu'il y a au moins 10 documents
            logging.info("Contenu du 1ème document: %s", documents[1].page_content)
            logging.info("Métadonnées du 1ème document: %s", documents[1].metadata)
        
        return documents

    except Exception as e:
        logging.error("Erreur lors du chargement des documents: %s", str(e))
        return []

def process_pdf(file):
    """
    Traite un fichier PDF, extrait le texte et le découpe en morceaux avec LangChain.
    
    Args:
        file: Fichier PDF uploadé
        
    Returns:
        List[Document]: Liste des documents LangChain découpés
    """
    try:
        # Lire le fichier PDF depuis la mémoire
        pdf_content = file.read()
        pdf_file = BytesIO(pdf_content)
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        logger.info(f"Nombre de pages dans le PDF : {len(pdf_reader.pages)}")
        
        # Extraire tout le texte du PDF
        text = ''
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + '\n'
                    logger.info(f"Page {page_num + 1} traitée : {len(page_text)} caractères")
                else:
                    logger.warning(f"Page {page_num + 1} est vide ou ne contient pas de texte extractible")
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'extraction de la page {page_num + 1} : {str(e)}")
                continue
        
        if not text.strip():
            raise Exception("Aucun texte extractible trouvé dans le PDF")
        
        logger.info("Texte extrait du PDF.")
        
        # Créer une liste de documents à partir du texte extrait
        documents = [Document(page_content=text, metadata={'filename': file.filename})]
        
        # Découper les documents en morceaux avec LangChain
        documents = chunk_text(documents, chunk_size=1024, chunk_overlap=100)
        
        logger.info(f"Extraction et découpage terminés : {len(documents)} morceaux créés")
        return documents
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF : {str(e)}")
        raise

def insert_to_chroma(documents, chroma_client):
    """
    Insère les documents LangChain traités dans ChromaDB.
    
    Args:
        documents: Liste des documents LangChain à insérer
        chroma_client: Client ChromaDB
    """
    try:
        # Configuration de l'embedding
        embeddings_model = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small", 
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Récupérer ou créer la collection
        collection = chroma_client.get_or_create_collection(
            name="Documents", 
            embedding_function=embeddings_model
        )
        
        # Préparer les données pour l'insertion
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            texts.append(doc.page_content)
            
            # Préparer les métadonnées
            metadata = doc.metadata.copy()
            metadata['chunk_index'] = i
            metadatas.append(metadata)
            
            # Générer un ID unique
            filename = doc.metadata.get('filename', 'unknown')
            doc_id = f"{filename}_{i}"
            ids.append(doc_id)
        
        # Insertion dans ChromaDB
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Insertion réussie : {len(texts)} morceaux ajoutés à ChromaDB")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion dans ChromaDB : {str(e)}")
        raise
