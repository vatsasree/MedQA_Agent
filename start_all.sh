#!/bin/bash

# Define paths and environment
ENV_PATH="/fsxvision_new/sreevatsa.s/ENVIRONMENTS/MedQA_Agent_Env/bin/activate"
LOG_FILE="/home/sreevatsa.s/MedQA_Agent/vllm.log"
APP_DIR="/home/sreevatsa.s/MedQA_Agent"

echo "================================================="
echo "  Starting MedQA Agent & vLLM Server via Tmux    "
echo "================================================="

# Environment Variables for NCCL (Fixes Segfaults on certain multi-GPU nodes)
export HF_TOKEN="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"
export VLLM_USE_DEEP_GEMM="0"
export NCCL_NET="Socket"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export NCCL_SHM_DISABLE="1"
export FI_PROVIDER="tcp"

# Kill existing sessions if they exist to prevent duplicates
tmux kill-session -t medqa_vllm 2>/dev/null
tmux kill-session -t medqa_app 2>/dev/null

# 1. Start vLLM Server in a new tmux session
echo "[1/3] Starting vLLM Server (Qwen/Qwen3.5-122B-A10B) in tmux session 'medqa_vllm'..."
tmux new-session -d -s medqa_vllm "export CUDA_VISIBLE_DEVICES='4,5,6,7' && export HF_TOKEN='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' && export VLLM_USE_DEEP_GEMM='0' && export NCCL_NET='Socket' && export NCCL_P2P_DISABLE='1' && export NCCL_IB_DISABLE='1' && export NCCL_SHM_DISABLE='1' && export FI_PROVIDER='tcp' && source $ENV_PATH && cd $APP_DIR && python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-122B-A10B --tensor-parallel-size 4 --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_xml --max-num-seqs 256 | tee $LOG_FILE 2>&1"

# 2. Wait for the server to be ready
echo "[2/3] Waiting for the vLLM server to become healthy (this takes a few minutes for a 122B model)..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 5
done
echo "      vLLM server is online and ready!"

# 3. Start the Streamlit Application in a new tmux session
echo "[3/3] Starting the Streamlit Dashboard in tmux session 'medqa_app'..."
tmux new-session -d -s medqa_app "source $ENV_PATH && cd $APP_DIR && streamlit run app.py"

echo "================================================="
echo "All components successfully started!"
echo "To monitor vLLM, run:       tmux a -t medqa_vllm"
echo "To monitor Streamlit, run:  tmux a -t medqa_app"
echo "================================================="
