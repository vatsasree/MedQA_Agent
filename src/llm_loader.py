import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

def get_llm(model_type="open-source", model_name="GPT-OSS-120B", temperature=0.0):
    """
    Modular LLM loader to easily switch between open-source and closed-source models.
    
    Args:
        model_type (str): "open-source", "openai", or "anthropic"
        model_name (str): The name of the model. Defaults to "GPT-OSS-120B".
        temperature (float): The temperature for sampling. Defaults to 0.0 for factual responses.
    """
    if model_type == "open-source":
        # Assuming the open source model is served via an OpenAI-compatible API
        # such as vLLM, Ollama, LM Studio, or a custom cluster endpoint.
        base_url = os.getenv("OPEN_SOURCE_API_BASE", "http://localhost:8000/v1")
        api_key = os.getenv("OPEN_SOURCE_API_KEY", "dummy-key")
        
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature
        )
        
    elif model_type == "openai":
        return ChatOpenAI(
            model=model_name, # e.g., "gpt-4o"
            temperature=temperature
        )
        
    elif model_type == "anthropic":
        return ChatAnthropic(
            model_name=model_name, # e.g., "claude-3-5-sonnet-20240620"
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

if __name__ == "__main__":
    print(f"Testing LLM loader configured for open-source...")
    llm = get_llm(model_type="open-source", model_name="GPT-OSS-120B")
    print("Successfully instantiated LLM object:")
    print(llm)
