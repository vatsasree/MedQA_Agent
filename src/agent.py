import os
from langchain_core.tools import create_retriever_tool
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from src.llm_loader import get_llm
from src.ingest import get_embeddings, DB_DIR

def build_agent(model_type="open-source", model_name="GPT-OSS-120B"):
    """
    Builds the ReAct Agent equipped with medical handbook and patient report tools.
    """
    # 1. Initialize Embeddings and LLM
    embeddings = get_embeddings()
    llm = get_llm(model_type=model_type, model_name=model_name)
    
    # 2. Connect to ChromaDB Collections
    if not os.path.exists(DB_DIR):
        print(f"Warning: ChromaDB directory '{DB_DIR}' not found. You need to run ingest.py first to populate the database.")
        
    handbook_vectorstore = Chroma(
        collection_name="medical_handbook",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    reports_vectorstore = Chroma(
        collection_name="patient_reports",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    # 3. Create Retrievers
    # Retrieve top 5 most relevant chunks
    handbook_retriever = handbook_vectorstore.as_retriever(search_kwargs={"k": 5})
    reports_retriever = reports_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 4. Create Tools
    handbook_tool = create_retriever_tool(
        handbook_retriever,
        "search_medical_handbook",
        "Searches and returns excerpts from the authoritative 6000-page medical handbook. Use this tool to look up medical guidelines, definitions, drug interactions, treatments, and standard clinical procedures."
    )
    
    reports_tool = create_retriever_tool(
        reports_retriever,
        "search_patient_reports",
        "Searches and returns excerpts from the patient medical reports. Use this tool to find specific patient history, test results, doctor notes, diagnoses, or current conditions."
    )
    
    tools = [handbook_tool, reports_tool]
    
    # 5. Create LangGraph ReAct Agent
    system_message = """You are a highly capable Medical Clinical Insight Engine (MedQA Agent). 
You have access to two primary sources of information via your tools:
1. 'search_medical_handbook': An authoritative Medical Handbook for clinical guidelines and medical facts.
2. 'search_patient_reports': Patient Medical Reports for specific patient information.

When answering a user's question, follow this reasoning process:
- If the question is about a specific patient or clinical case, first search the patient reports to gather the facts.
- If the question asks for medical advice, procedures, treatments, or definitions related to what you found in the patient reports (or a general medical question), search the medical handbook.
- Combine the information from both sources to provide a comprehensive, accurate clinical insight.
- Always cite your sources clearly (e.g., 'According to the medical handbook...' or 'Based on the patient's MRI report...').
- If you cannot find the answer in the provided documents, state that clearly instead of guessing."""

    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    return agent_executor

if __name__ == "__main__":
    print("Building MedQA Agent...")
    try:
        agent = build_agent()
        print("Agent built successfully.")
    except Exception as e:
        print(f"Error building agent: {e}")
