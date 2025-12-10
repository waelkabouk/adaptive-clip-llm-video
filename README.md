# CLIP-LLM Video Understanding Pipeline

A modular zero-shot video understanding pipeline using CLIP-based semantic de-duplication and LangGraph orchestration for efficient VideoQA, captioning, retrieval, and temporal grounding.

## Features

- **Multi-task Support**: VideoQA, captioning, retrieval, temporal grounding
- **Semantic De-duplication**: FAISS-powered frame deduplication using CLIP embeddings
- **Memory-Aware**: Optimized for 4GB VRAM GPUs with batch size autoscaling
- **API-First Inference**: GPT-4V, Claude, Gemini support with cost tracking
- **LangGraph Orchestration**: Flexible task routing and context building
- **Chainlit UI**: Interactive web interface for video upload and querying

## Architecture

```
Video → Frame Extraction → CLIP Encoding → FAISS Dedup → LangGraph Pipeline → LLM API → Response
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Or configure them in the Chainlit UI.

### Running the Web UI

```bash
chainlit run chainlit_app.py
```

### Running Evaluation

```bash
python scripts/run_benchmark.py --config configs/api_first.yaml
```

## Project Structure

```
├── configs/               # Hydra configuration files
│   ├── api_first.yaml    # API-based inference config
│   ├── local_tiny.yaml   # Local tiny VLM config
│   └── tasks/            # Task-specific configs
├── src/
│   ├── video/            # Video loading and sampling
│   ├── encoders/         # CLIP encoder abstraction
│   ├── dedup/            # FAISS-based deduplication
│   ├── pipeline/         # LangGraph orchestration
│   ├── llm/              # LLM API clients
│   └── evaluation/       # Metrics and benchmarking
├── chainlit_app.py       # Chainlit web UI
└── scripts/              # Utility scripts
```

## Hardware Requirements

- **Minimum**: 4GB VRAM GPU (GTX 1650 Ti level)
- **Recommended**: 8GB+ VRAM for faster processing
- **CPU-only**: Supported but slower

## License

MIT License

