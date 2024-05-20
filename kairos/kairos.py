import os
import glob
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load PDF files from directory
pdf_dir = "/app/pdf_files"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

# Load PDF files
data = []
for pdf_file in pdf_files:
    loader = UnstructuredPDFLoader(file_path=pdf_file)
    data.extend(loader.load())

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)

# LLM from Ollama
local_model = "phi3"
llm = ChatOllama(model=local_model)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the RAG chain
chain = (
    {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
query = "Based on the Declaration of Conformity issued in July 2022, what we can say?"
result = chain.invoke(query)
print(result)
