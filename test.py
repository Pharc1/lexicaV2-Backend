import chromadb

chroma_client = chromadb.HttpClient(
    host="172.20.10.8",
    port=8000
)
print(chroma_client.heartbeat())