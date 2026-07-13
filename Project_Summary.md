# MedQA Agent: Project Context & Resume Guide

## 1. Is this a good project for a resume?
**Absolutely. It is exceptionally strong.** 

95% of AI projects on resumes today are simply "wrappers"—students or junior developers sending basic text to the OpenAI API and printing the result. This project stands out completely because it demonstrates **Deep AI Systems Engineering**. 

You didn't just call an API; you:
1. Deployed an absolutely massive, state-of-the-art open-weights model (**122 Billion parameters**) locally.
2. Handled **Distributed Inference** (Tensor Parallelism across 4 GPUs).
3. Debugged deep, complex infrastructure issues (NCCL networking segfaults, CUDA Mamba cache memory constraints).
4. Handled complex Agentic orchestration (LangGraph state machines) rather than simple chat loops.
5. Bridged the gap between open-source model formats (XML tool calling) and enterprise frameworks (LangChain).

### How to portray this on your resume
Depending on what role you are targeting (AI Engineer, ML Ops, Software Engineer), here are highly effective bullet points you can use:

> **AI Systems Engineer - MedQA Autonomous Agent**
> * **Architected and deployed** an autonomous Medical QA Agent (LangGraph/Streamlit) powered by a locally hosted Qwen3.5-122B model on a multi-GPU cluster.
> * **Engineered high-throughput distributed inference** using `vLLM` with tensor parallelism; diagnosed and resolved complex PyTorch/NCCL shared-memory segfaults to ensure stable execution.
> * **Optimized GPU memory utilization** by configuring CUDA graph capture limits and KV cache sequences (`max_num_seqs`), preventing Out-Of-Memory (OOM) crashes across 4 hardware accelerators.
> * **Implemented robust tool-calling integration** by mapping native XML outputs from open-weight models to strict JSON schemas expected by LangChain execution logic.
> * **Automated infrastructure deployment** by writing robust bash scripts and `tmux` multiplexing pipelines to manage decoupled frontend and backend microservices safely.

---

## 2. Full Context for Claude (Future Brainstorming)
*Copy and paste everything below this line to Claude to instantly bring it up to speed on what we built.*

### Project Overview
**MedQA** is an autonomous Medical Clinical Insight Engine. It allows users to ask complex medical questions. The agent intelligently decides whether to query a database of clinical guidelines (Medical Handbooks) or query specific patient history (Patient Reports) to formulate highly accurate, context-aware answers.

### Tech Stack
* **Frontend:** Streamlit (`app.py`)
* **Agent Orchestration:** LangChain & LangGraph (`src/agent.py`)
* **Local Inference Server:** `vLLM` (OpenAI-compatible API Server)
* **LLM:** `Qwen/Qwen3.5-122B-A10B`
* **Hardware:** 4x NVIDIA GPUs (Tensor Parallelism)
* **Automation:** Bash (`start_all.sh`) & `tmux`

### Architecture & Workflows
1. **The Backend (`vLLM`):** We run the 122B parameter model locally on port 8000 using `vllm.entrypoints.openai.api_server`. The server is decoupled from the frontend to prevent model reloading.
2. **The Frontend (`Streamlit`):** Provides the chat UI. It communicates with the local vLLM server via LangChain's `ChatOpenAI` wrapper.
3. **The Agent (`LangGraph`):** Uses a ReAct/Tool-calling loop. It is equipped with Python functions (`search_medical_handbook`, `search_patient_reports`) to pull real data into its context window.

### Key Technical Hurdles Overcome (Context for Claude)
If we build new features, Claude must be aware of the following environment constraints and fixes we already implemented:

1. **NCCL Segfaults on Multi-GPU:** 
   * *Issue:* `vLLM` crashed with deep PyTorch/NCCL segmentation faults when attempting to shard the model.
   * *Fix:* We explicitly disabled buggy hardware interconnects by setting `NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, and `NCCL_SHM_DISABLE=1`.
2. **Tmux Environment Inheritance:**
   * *Issue:* The NCCL fixes were ignored because `tmux new-session` does not inherit variables from bash scripts if a tmux daemon is already running.
   * *Fix:* We injected the `export` commands directly into the string executed by `tmux`.
3. **Open-Source Tool Calling (400 BadRequest):**
   * *Issue:* LangGraph failed to execute tools because Qwen generates XML tags (`<tool_call><function=...>`) instead of JSON. 
   * *Fix:* We launched vLLM with `--enable-auto-tool-choice` and `--tool-call-parser qwen3_xml` so the server natively intercepts the XML and translates it to OpenAI-compliant JSON for LangChain.
4. **Mamba / KV Cache OOM:**
   * *Issue:* Restricting the 122B model to 4 GPUs (from 8) caused CUDA graph capture to fail because the default 1024 concurrent sequences required more Mamba cache blocks than available in the remaining VRAM.
   * *Fix:* We added `--max-num-seqs 256` and isolated the GPUs using `CUDA_VISIBLE_DEVICES='4,5,6,7'`.

---

## 3. Future Extensions & Tooling Ideas
If you want to take this project from a "strong portfolio piece" to an "Enterprise-Grade Architecture", here are the best directions to extend it:

### A. Advanced Medical Tooling
You can easily add new Python functions and bind them to the LangGraph agent to give it superpowers:
* **`search_pubmed_api(query)`**: Hook the agent up to the PubMed or ClinicalTrials.gov APIs. If the local handbook doesn't have the answer, the agent autonomously decides to search the internet for the absolute latest clinical trials and research papers.
* **`generate_soap_note(patient_id, findings)`**: A tool that allows the agent to take its conversation with you, format it into a professional SOAP (Subjective, Objective, Assessment, Plan) note, and save it as a PDF or text file.
* **`fetch_fhir_record(patient_id)`**: Transition from reading static PDFs to querying a dummy FHIR (Fast Healthcare Interoperability Resources) REST API, which is the gold standard for modern hospital EHR systems.

### B. Multi-Modal Vision (VLM) Integration
* **`analyze_medical_image(image_path)`**: Upgrade the agent to handle Vision. If a user uploads an X-Ray, MRI, or a scanned handwritten prescription, the agent can pass it to an open-source VLM (like `Qwen-VL` or `LLaVA`) to extract the text or summarize the anomalies, and then factor that into its final answer.

### C. Multi-Agent Architecture
Transition from a single "do-it-all" agent to a **Hierarchical Multi-Agent System**:
* **Triage Agent:** Analyzes the user's prompt and routes it to the correct specialist.
* **Researcher Agent:** Specifically tasked with pulling and synthesizing dense medical documents.
* **Reviewer Agent:** A strict safety-guardrail agent that intercepts the Researcher's draft and ensures it doesn't contain hallucinated medications or dangerous advice before finally showing it to the user.

### D. Persistent Memory
Right now, the agent likely forgets everything if you refresh the Streamlit page.
* **Implementation:** Add PostgreSQL or a vector database (like Chroma/Milvus) to store the conversation history and patient context. This allows the agent to remember that you asked about "John Doe's heart rate" three days ago.
