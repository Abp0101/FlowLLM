# FlowLLM 🚦

> AI-powered traffic light optimization using a locally-run LLM to reduce vehicle wait times at intersections.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![SUMO](https://img.shields.io/badge/SUMO-1.24.0-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.1_8B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Results

| Metric | Fixed-Cycle Baseline | LLM Controller | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Wait Time | 14.4s | 12.4s | **14% faster** |
| Avg Queue Length | 91.6 | 69.5 | **24% shorter** |
| Vehicles Processed | 278 | 288 | **+10 vehicles** |

## Overview

FlowLLM replaces traditional fixed-cycle traffic lights with a locally-run LLM (Llama 3.1 8B via Ollama) that dynamically adjusts signal timing based on real-time queue data. The LLM receives the current intersection state every 10 simulation steps and decides whether to switch phases and for how long.

The entire system runs locally — no API costs, no cloud dependency.

## How It Works

1. **SUMO** simulates a 4-way intersection with randomised vehicle flows over 600 seconds
2. **TraCI** (Python API) connects to SUMO and reads live queue lengths and wait times
3. **Ollama** serves Llama 3.1 8B locally, receiving intersection state as a structured prompt
4. The LLM responds with a JSON decision: switch phase or keep current, and for how long
5. TraCI applies the decision to the traffic light in real time
6. Metrics are logged and compared against a fixed-cycle baseline

## Project Structure
```
FlowLLM/
├── Simulation/          # SUMO network files and config
├── Agent/
│   ├── baseline_controller.py   # Fixed-cycle controller
│   └── llm_controller.py        # LLM-powered controller
├── Data/
│   ├── baseline_metrics.csv
│   └── llm_metrics.csv
├── Dashboard/
│   └── app.py           # Streamlit visualisation
├── Evaluation/
├── requirements.txt
└── LICENSE
```

## Prerequisites

**1. Install SUMO**
Download and install from https://sumo.dlr.de/docs/Downloads.php

**2. Install Ollama**
Download from https://ollama.com then pull the model:
```bash
ollama pull llama3.1:8b
```

## Setup

**Clone the repo:**
```bash
git clone https://github.com/Abp0101/FlowLLM.git
cd FlowLLM
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

## Running

**Baseline controller:**
```bash
python Agent/baseline_controller.py
```

**LLM controller:**
```bash
python Agent/llm_controller.py
```

**Dashboard:**
```bash
streamlit run Dashboard/app.py
```

## Key Design Decisions

- **Local LLM only** — keeps the project fully free and reproducible by anyone with a decent GPU
- **Prompt engineering matters** — v1 of the prompt performed worse than baseline; v2 with explicit traffic rules and JSON formatting beat it on all metrics
- **Stepped simulation** — SUMO runs in lockstep with the LLM rather than real time, avoiding latency issues

## Future Work

- Extend to a multi-intersection grid network
- Fine-tune a smaller model specifically on traffic control data
- Add reinforcement learning to improve decisions over time
- Test with real-world traffic flow data from open datasets

## Tech Stack

SUMO · Python · TraCI · Ollama · Llama 3.1 8B · Streamlit · Plotly · Pandas