import os
import glob
import http.server
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import chromadb

# Load PDF files from directory
pdf_dir = "../project/PKM150_docs/"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
LOCAL_DB = 'pk_vectordb_new'
COLLECTION = 'pk_collection_new'
# Load PDF files

# Chroma client
#collection = persistent_client.get_or_create_collection(COLLECTION)
persistent_client = chromadb.PersistentClient(LOCAL_DB)
try:
    collection = persistent_client.get_collection(COLLECTION)
    vector_db = Chroma(
            client=persistent_client,
            collection_name=COLLECTION,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            )
except ValueError as e:
    print(f"{e}")
    data = []
    for pdf_file in pdf_files:
        print(f"{pdf_file}")
        loader = UnstructuredPDFLoader(file_path=pdf_file)
        data.extend(loader.load())

    # Split and chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    collection = persistent_client.create_collection(COLLECTION)

    # Passing Chroma Client to Langchain
    vector_db = Chroma.from_documents(
        client=persistent_client,
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True), 
        collection_name=COLLECTION
    )

# LLM from Ollama
local_model = "phi3"
llm = ChatOllama(model=local_model)

# RAG prompt
template = """Answer the question based ONLY on the following context: {context} Question: {question} """
#template = """ fake the following question: How to fix a Tritium fast charger RT50 with a problem in the pump  in ONLY 4 STEPS. pl,ease be concise and DO NOT add any extra note at the end of the steps."""
prompt = ChatPromptTemplate.from_template(template)

# Define the RAG chain
chain = (
    {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#def ask_question(query):
#    result = chain.invoke(query)
#    return result
#
#class RequestHandler(http.server.BaseHTTPRequestHandler):
#    def do_POST(self):
#        content_length = int(self.headers['Content-Length'])
#        post_data = self.rfile.read(content_length)
#        query = json.loads(post_data)['query']
#        answer = ask_question(query)
#        self.send_response(200)
#        self.send_header('Content-type', 'application/json')
#        self.end_headers()
#        self.wfile.write(json.dumps({'answer': answer}).encode())

def ask_question(query):
    # Retrieve context from vector database
    context = vector_db.as_retriever().get_relevant_documents(query)
    print("Retrieved context:")
    for doc in context:
        print(doc)  # Debugging: print each retrieved document


    # Generate the answer without RAG (directly using the language model)
    llm_result = llm.predict(query)
    print(f"Generated answer without RAG: {llm_result}")  # Debugging: print the LLM-only answer

    # Generate the answer using the RAG chain
    rag_result = chain.invoke(query)
    print(f"Generated answer with RAG: {rag_result}")  # Debugging: print the RAG answer
    return {
        "with_RAG": rag_result,
        "without_RAG": llm_result
    }

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        query = json.loads(post_data)['query']
        answers = ask_question(query)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(answers).encode())

if __name__ == '__main__':
    server_address = ('0.0.0.0', 5000)
    httpd = http.server.HTTPServer(server_address, RequestHandler)
    print(f'Starting server at http://{server_address[0]}:{server_address[1]}')
    httpd.serve_forever()


