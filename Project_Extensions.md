# Project Expansion Ideas

## Part 1: Extending MedQA Agent — Multi-Agent Architecture

Your current MedQA agent uses a **single ReAct agent** with two tools. That's a great starting point, but to make this truly impressive, you want to move towards a **Multi-Agent System** where specialized agents collaborate, each with their own personality, tools, and responsibilities.

### Proposed Architecture: MedQA Multi-Agent Team

```
                    ┌──────────────────┐
                    │   Triage Agent   │  ← Entry point. Classifies the query.
                    │  (Router/Planner)│
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
  ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
  │ Clinical Guide  │ │   Patient    │ │   Drug Safety    │
  │    Researcher   │ │   Analyst    │ │    Specialist    │
  │                 │ │              │ │                  │
  │ Tools:          │ │ Tools:       │ │ Tools:           │
  │ • handbook_search│ │ • patient_db │ │ • drug_interaction│
  │ • pubmed_api    │ │ • lab_results│ │   _checker       │
  │ • guideline_    │ │ • vitals_    │ │ • fda_recall_api │
  │   summarizer    │ │   timeline   │ │ • contraindication│
  └────────┬────────┘ └──────┬───────┘ │   _lookup        │
           │                 │         └────────┬─────────┘
           └────────────────┬┘                  │
                            ▼                   │
                  ┌─────────────────┐           │
                  │  Safety Reviewer│ ◄─────────┘
                  │   (Guardrail)   │
                  │                 │
                  │ • Checks for    │
                  │   hallucinations│
                  │ • Validates drug│
                  │   names exist   │
                  │ • Adds disclaimers│
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Report Writer  │
                  │                 │
                  │ • Formats SOAP  │
                  │   notes         │
                  │ • Generates PDF │
                  │ • Structures    │
                  │   citations     │
                  └─────────────────┘
```

### Agent Descriptions

| Agent | Role | Tools | Why It's Impressive |
|-------|------|-------|---------------------|
| **Triage Agent** | Classifies incoming queries and routes to the correct specialist(s). Decides if the question is about general guidelines, a specific patient, or drug safety. | `classify_intent()` | Shows you understand intelligent routing and orchestration. |
| **Clinical Guideline Researcher** | Deep-dives into medical handbooks, PubMed, and clinical guidelines to find evidence-based answers. | `search_medical_handbook()`, `search_pubmed_api()`, `summarize_guidelines()` | RAG + external API integration. |
| **Patient Analyst** | Focuses exclusively on patient-specific data — lab results, vitals history, imaging reports. | `search_patient_reports()`, `get_lab_results()`, `plot_vitals_timeline()` | Shows structured data querying and visualization. |
| **Drug Safety Specialist** | Checks for drug-drug interactions, contraindications, and FDA recalls before any medication is recommended. | `check_drug_interactions()`, `lookup_contraindications()`, `search_fda_recalls()` | Critical safety layer — extremely relevant in healthcare AI. |
| **Safety Reviewer** | A strict guardrail agent that reviews the final draft. Ensures no hallucinated drug names, validates dosage ranges, adds mandatory disclaimers. | `validate_drug_exists()`, `check_dosage_range()`, `add_disclaimer()` | Shows you understand AI safety and responsible deployment. |
| **Report Writer** | Takes the validated findings and formats them into professional clinical documents (SOAP notes, summaries with citations). | `generate_soap_note()`, `format_citation()`, `export_pdf()` | Shows structured output generation. |

### How to Implement in LangGraph
LangGraph natively supports multi-agent graphs. Instead of a single `StateGraph` with one agent node, you create multiple agent nodes and connect them with conditional edges:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(MedQAState)

# Add agent nodes
graph.add_node("triage", triage_agent)
graph.add_node("researcher", clinical_researcher_agent)
graph.add_node("patient_analyst", patient_analyst_agent)
graph.add_node("drug_safety", drug_safety_agent)
graph.add_node("safety_reviewer", safety_reviewer_agent)
graph.add_node("report_writer", report_writer_agent)

# Triage routes to the right specialist(s)
graph.add_conditional_edges("triage", route_to_specialist, {
    "guidelines": "researcher",
    "patient": "patient_analyst",
    "drug_check": "drug_safety",
})

# All specialists feed into safety review
graph.add_edge("researcher", "safety_reviewer")
graph.add_edge("patient_analyst", "safety_reviewer")
graph.add_edge("drug_safety", "safety_reviewer")

# Safety review feeds into report writer
graph.add_edge("safety_reviewer", "report_writer")
graph.add_edge("report_writer", END)

graph.set_entry_point("triage")
app = graph.compile()
```

---

## Part 2: CarnaticLLM — A Carnatic Music Knowledge Agent

This is a **phenomenal** and highly unique project idea. Nobody else will have this on their resume. The intersection of NLP + Multilingual Indian Languages + Music Theory is incredibly niche and impressive.

### What You Have
- Scraped Kriti lyrics in multiple Indian languages (Sanskrit, Telugu, Tamil, Kannada, etc.)
- Word-by-word meanings and explanations
- (Presumably) metadata like Raga, Tala, Composer, etc.

### What You Can Build

#### Architecture: CarnaticLLM Multi-Agent System

```
                    ┌──────────────────┐
                    │   Query Router   │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐
  │  Kriti Scholar│  │  Raga Analyst │  │ Language Expert  │
  │              │  │               │  │                  │
  │ Tools:       │  │ Tools:        │  │ Tools:           │
  │ • search_    │  │ • raga_lookup │  │ • translate_     │
  │   kriti_db   │  │ • find_kritis │  │   lyrics         │
  │ • get_meaning│  │   _by_raga    │  │ • get_word_      │
  │ • get_sahitya│  │ • raga_        │  │   meanings       │
  │ • composer_  │  │   comparison  │  │ • transliterate  │
  │   biography  │  │ • suggest_    │  │ • explain_       │
  │              │  │   similar_raga│  │   sandhi_rules   │
  └──────────────┘  └───────────────┘  └─────────────────┘
```

### Possible Tools (Python Functions)

| Tool | Description | Example Query |
|------|-------------|---------------|
| `search_kriti_db(query)` | RAG search across all Kriti lyrics and meanings | *"Find kritis that talk about devotion to Rama"* |
| `get_word_meanings(kriti_name, language)` | Returns word-by-word breakdown | *"Give me the word-by-word meaning of Vatapi Ganapatim"* |
| `get_sahitya(kriti_name)` | Returns full lyrics in requested script | *"Show me the lyrics of Nagumomu in Telugu script"* |
| `translate_lyrics(kriti_name, target_lang)` | Translates lyrics to English or another Indian language | *"Translate Brova Bharama to English"* |
| `raga_lookup(raga_name)` | Returns arohana, avarohana, janya/melakarta info, mood | *"Tell me about Raga Kalyani"* |
| `find_kritis_by_raga(raga_name)` | Lists all kritis composed in a specific raga | *"Which famous kritis are in Raga Todi?"* |
| `find_kritis_by_composer(composer)` | Lists all kritis by a specific vaggeyakara | *"List all kritis by Muthuswami Dikshitar"* |
| `raga_comparison(raga1, raga2)` | Compares two ragas — shared swaras, mood differences | *"What is the difference between Shankarabharanam and Kalyani?"* |
| `suggest_similar_ragas(raga_name)` | Finds related ragas by swaras or mood | *"Which ragas sound similar to Bhairavi?"* |
| `explain_sandhi_rules(word)` | Explains Sanskrit sandhi (word fusion) in lyrics | *"Why does 'Vatapi' become 'Vatapim' in the kriti?"* |
| `composer_biography(name)` | Returns historical context about vaggeyakaras | *"Tell me about the life of Tyagaraja"* |

### Sample Conversations

**User:** *"I want to learn a kriti in Raga Mohanam. Something by Tyagaraja that's beginner-friendly."*
**Agent (internally):**
1. Calls `find_kritis_by_raga("Mohanam")` → gets list
2. Filters by `find_kritis_by_composer("Tyagaraja")` → narrows down
3. Calls `get_sahitya("Nanu Palimpa")` → fetches lyrics
4. Calls `get_word_meanings("Nanu Palimpa", "english")` → word breakdown
5. Returns a beautifully formatted response with lyrics, meanings, and raga context.

**User:** *"What is the philosophical meaning behind Sri Ramam Raghukula Tilakam?"*
**Agent:** Searches the kriti database, pulls word-by-word meanings, and synthesizes a philosophical interpretation drawing from the Sahitya annotations.

**User:** *"Compare Raga Kambhoji and Raga Shankarabharanam"*
**Agent:** Calls `raga_comparison()` and produces a structured table showing shared swaras, differences, mood, and example kritis in each.

### Why This is a Resume Powerhouse
- **Multilingual NLP:** Working with Sanskrit, Telugu, Tamil, Kannada simultaneously is a massive differentiator.
- **Domain-Specific RAG:** Building a retrieval system over a niche cultural corpus (not just generic Wikipedia).
- **Cultural AI:** The intersection of AI and Indian classical music is virtually unexplored in the industry.
- **Agentic Architecture:** Multi-tool, multi-agent setup shows systems-level thinking.

### Tech Stack Suggestion for CarnaticLLM
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit (reuse your MedQA skills) |
| Agent | LangGraph (reuse your MedQA architecture) |
| Vector DB | ChromaDB or Qdrant (for Kriti embeddings) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (supports Indian languages) |
| LLM | Qwen3.5-122B (reuse your vLLM setup) or a smaller model like Qwen2.5-7B for faster iteration |
| Data Storage | SQLite or PostgreSQL for structured kriti metadata (raga, tala, composer, language) |

### Data Pipeline
```
Raw Scraped Data (JSON/CSV)
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  ← Clean, normalize scripts, tag languages
│  & Structuring   │  ← Extract: kriti_name, raga, tala, composer, lyrics, meanings
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ SQLite │ │ ChromaDB │  ← Embed lyrics + meanings for semantic search
│ (meta) │ │ (vectors)│
└────────┘ └──────────┘
```
