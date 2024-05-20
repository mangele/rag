import os
import glob
import json
import http.server
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import chromadb
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Directory for persisting vector store
VECTOR_STORE_DIR = "./vector_store"

# Initiate persistent Chroma client
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

# Define the embedding function using OllamaEmbeddings
class EmbeddingFunction:
    def __init__(self):
        self.model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

    def __call__(self, input):
        # Use embed_documents method for multiple documents
        return self.model.embed_documents(input)

embedding_function = EmbeddingFunction()

# Create or get collection
collection_name = "local-rag"
try:
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    print(f"Loaded existing collection: {collection_name}")
except ValueError:
    print(f"Creating new collection: {collection_name}")

    # Load PDF files from directory
    pdf_dir = "../../project/PKM150_docs/"
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    # Load PDF files
    data = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")
        loader = UnstructuredPDFLoader(file_path=pdf_file)
        data.extend(loader.load())

    # Split and chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # Create collection and add documents
    collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        ids=[str(i) for i in range(len(chunks))]
    )
    print(f"Created and persisted new collection: {collection_name}")

class CombinedServerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_data = json.loads(post_data)

        if 'query' in request_data:
            self.handle_query(request_data['query'])
        elif 'prompt' in request_data:
            self.handle_completion(request_data['prompt'], request_data['n_predict'])

    def handle_query(self, query):
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        response_data = {'results': [result for result in results['documents']]}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())



    def handle_completion(self, prompt, n_predict):
        url = "http://localhost:8080/completion"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt, "n_predict": n_predict, 'cache_prompt': False}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(response.content)

if __name__ == '__main__':
    server_address = ('0.0.0.0', 5000)
    httpd = HTTPServer(server_address, CombinedServerHandler)
    print(f'Starting combined server at http://{server_address[0]}:{server_address[1]}')
    httpd.serve_forever()

