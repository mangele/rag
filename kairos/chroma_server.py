import os
import glob
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import http.server
import json
from langchain.document_loaders import UnstructuredFileLoader
import chromadb
# Load PDF files from directory
#pdf_dir = "/app/pdf_files"
pdf_dir = "../project/PKM150_docs/"
#pdf_dir = "PK150_docs_samples/"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))


## Load C++ and header files from directory
#code_dir = "/home/mgl/RAG/kairos/llama.cpp/examples/server/"
#pdf_files = [os.path.join(code_dir, f) for f in os.listdir(code_dir) if f.endswith(('.cpp', '.hpp'))]

# Load PDF files
documents = []
for pdf_file in pdf_files:
    print(f"{pdf_file}")
    loader = UnstructuredPDFLoader(file_path=pdf_file)
#    loader = UnstructuredFileLoader(file_path=pdf_file)
    documents.extend(loader.load())

# Split and chunk text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)


client = chromadb.Client()
if client.list_collections():
    pk_collection = client.create_collection("pk_docs")
else:
    print("Collection already exists")

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    persist_directory='./pk_docs_db'
)
vector_db.persist()

class ChromaServerHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        query = json.loads(post_data)['query']
        results = vector_db.similarity_search(query, k=3)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'results': [result.page_content for result in results]}).encode())

if __name__ == '__main__':
    server_address = ('0.0.0.0', 5000)
    httpd = http.server.HTTPServer(server_address, ChromaServerHandler)
    print(f'Starting Chroma server at http://{server_address[0]}:{server_address[1]}')
    httpd.serve_forever()
