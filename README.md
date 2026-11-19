# OMNIBRAIN AI

### Building Intelligence from the Ground Up

OMNIBRAIN AI is an independent foundation-model research project focused on its own neural architecture, tokenizer, dataset engineering, training pipeline, model checkpoints, and local inference engine.

## Project Objectives
- Develop an independent decoder-only Transformer architecture.
- Build and evaluate custom tokenization systems.
- Engineer curated datasets with provenance and licensing records.
- Train model weights using PyTorch.
- Implement local inference and checkpoint management.

## Technology Stack
- Frontend: React, TypeScript, Vite
- Backend: Node.js, Express
- Machine Learning: Python, PyTorch
- Architecture: Custom Transformer
- Communication: Local Node.js–Python process bridge

## Current Research Status
OMNIBRAIN AI is an experimental foundation-model project. Initial CPU pretraining experiments have been completed and documented. The current model is small and is not production-ready.

## Prerequisites
- Node.js
- Python 3.11
- Git

## Installation

### 1. Clone
```bash
git clone https://github.com/AKULAHEMANTH44/omnibrain-ai.git
cd omnibrain-ai
```

### 2. Install Node dependencies
```bash
npm install
```

### 3. Create Python environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install PyTorch
Install a compatible CPU version of PyTorch using the official PyTorch installation instructions.

### 5. Start
```bash
npm run dev
```

Open the local URL displayed in the terminal.

## Project Structure
```text
omnibrain-ai/
├── checkpoints/
├── datasets/
├── server/
├── src/
├── package.json
└── server.ts
```

## Research Principles
- Independent model development
- Reproducible experiments
- Transparent evaluation
- Dataset provenance and licensing awareness
- Local execution
- No fabricated training results or capability claims

## License
A repository license will be added after reviewing the licensing and redistribution requirements for the code, datasets, and model artifacts.

---
OMNIBRAIN AI — Independent Foundation Model Research
