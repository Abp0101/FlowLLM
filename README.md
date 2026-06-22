# FlowLLM 🚦

> AI-powered traffic light optimization using a locally-run LLM to reduce vehicle wait times at intersections.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![SUMO](https://img.shields.io/badge/SUMO-1.24.0-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.1_8B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Why this is an AI control agent

FlowLLM is a closed-loop AI control agent for traffic signal optimisation. It observes the simulated intersection state, asks a locally hosted LLM to choose a signal-control action, applies that action through TraCI, and measures the resulting traffic performance against a fixed-cycle baseline.

Agent loop:

1. SUMO simulates traffic flow at a four-way intersection
2. TraCI reads live queue lengths, waiting times and signal state
3. The LLM receives the current traffic state as a structured prompt
4. The model returns a JSON decision: keep or switch phase, plus duration
5. TraCI applies the decision to the traffic light
6. Metrics are logged and compared against the fixed-cycle controller

This demonstrates autonomous decision-making within a simulation, environment feedback, structured model outputs and metric-driven evaluation.

## Prototype results

| Metric | Fixed-Cycle Baseline | LLM Controller | Improvement |
|--------|---------------------|----------------|-------------|
| Avg Wait Time | 14.4s | 12.4s | **14% faster** |
| Avg Queue Length | 91.6 | 69.5 | **24% shorter** |
| Vehicles Processed | 278 | 288 | **+10 vehicles** |

These figures are results from the current prototype comparison, not evidence of performance in a physical traffic system. The saved 600-step CSVs reproduce the average waiting-time and queue-length figures. Vehicle throughput is calculated by the controller during a run and printed in its summary, but is not yet persisted in the metrics CSV.

## Evaluation focus

- The fixed-cycle controller provides the baseline using a 70-second cycle: 31 seconds green and 4 seconds yellow for each corridor.
- The LLM controller is evaluated using average waiting time, average queue length and vehicles processed during the simulation.
- Per-step metrics are saved to CSV, making the waiting-time and queue-length comparison inspectable and reproducible for the included simulation configuration.
- Current results should be treated as prototype results, not as a production traffic-control claim.
- Future evaluation should use repeated random seeds, confidence intervals and comparison against adaptive non-LLM baselines.

See [Architecture](docs/architecture.md), [Evaluation](docs/evaluation.md), [Limitations](docs/limitations.md) and [Roadmap](docs/roadmap.md) for the detailed system design and evaluation boundaries.

## Overview

FlowLLM compares a fixed-cycle traffic-light controller with a locally run LLM controller (Llama 3.1 8B via Ollama) that adjusts simulated signal timing based on current queue data. The LLM receives the intersection state every 10 simulation steps and decides whether to switch phases and for how long.

The default setup runs model inference locally through Ollama, without a hosted model API.

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

- **Local LLM inference** — keeps prompts and model execution on the local Ollama service by default.
- **Structured decisions** — the prompt requests a JSON action and duration; the controller validates actions, clamps durations and falls back to a keep decision when inference fails.
- **Stepped simulation** — SUMO advances in simulation steps, so the prototype does not establish real-time physical control performance.

## Future Work

The prioritised engineering and evaluation work is tracked in the [roadmap](docs/roadmap.md).

## Tech Stack

SUMO · Python · TraCI · Ollama · Llama 3.1 8B · Streamlit · Plotly · Pandas
