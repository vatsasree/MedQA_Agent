import os
import pytest
from src.llm_loader import get_llm

def test_get_llm_openai_initialization():
    """Test that the factory can initialize an OpenAI LLM correctly."""
    llm = get_llm(model_type="openai", model_name="gpt-4", temperature=0.5)
    assert llm.model_name == "gpt-4"
    assert llm.temperature == 0.5

def test_get_llm_opensource_initialization(monkeypatch):
    """Test that the open-source model initializes with correct environment variables."""
    monkeypatch.setenv("OPEN_SOURCE_API_BASE", "http://test-server:8000/v1")
    monkeypatch.setenv("OPEN_SOURCE_API_KEY", "test-key")
    
    llm = get_llm(model_type="open-source", model_name="Qwen-Test", temperature=0.1)
    assert llm.model_name == "Qwen-Test"
    assert llm.temperature == 0.1
    # Note: langchain_openai ChatOpenAI properties might vary by version, 
    # but model_name and temperature are standard.

def test_get_llm_invalid_type():
    """Test that an invalid model type raises a ValueError."""
    with pytest.raises(ValueError):
        get_llm(model_type="invalid-type")
