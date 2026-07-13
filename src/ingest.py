import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
HANDBOOK_DIR = os.path.join(DATA_DIR, "handbook")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def get_embeddings():
    """
    Returns the embedding model.
    Using HuggingFace local embeddings by default for privacy with medical data.
    """
    # BAAI/bge-small-en-v1.5 is a very strong and lightweight open-source embedding model.
    model_name = "BAAI/bge-small-en-v1.5" 
    # Change to 'cuda' if GPU is heavily available, 'cpu' is fine for small/medium workloads
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def ingest_documents(source_dir, collection_name):
    """
    Ingest PDFs from source_dir into a Chroma collection.
    """
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist.")
        return

    pdf_files = glob.glob(os.path.join(source_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {source_dir}")
        return
    
    docs = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        # Load the document (PyPDFLoader natively handles searchable PDFs)
        loaded_docs = loader.load()
        # Ensure metadata contains info about the source
        for doc in loaded_docs:
            doc.metadata["source_type"] = collection_name
        docs.extend(loaded_docs)
        
    print(f"Loaded {len(docs)} pages from {source_dir}. Splitting text...")
    
    # Medical texts often have complex sentences and lists.
    # 1000 tokens with 200 overlap is a standard strong baseline for RAG.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks. Generating embeddings and storing in ChromaDB...")
    
    embeddings = get_embeddings()
    
    # Create or update vector store locally
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR
    )
    print(f"Successfully ingested into collection '{collection_name}' at {DB_DIR}")
    return vectorstore

if __name__ == "__main__":
    print("--- MedQA Data Ingestion Pipeline ---")
    
    print("Ingesting Medical Handbooks...")
    ingest_documents(HANDBOOK_DIR, "medical_handbook")
    
    print("\nIngesting Patient Reports...")
    ingest_documents(REPORTS_DIR, "patient_reports")
    
    print("\nIngestion Process Complete!")
