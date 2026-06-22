# FlowLLM architecture

FlowLLM is a simulation-based control loop that compares two traffic-light policies under the same SUMO network configuration: a deterministic fixed-cycle controller and a controller whose actions are proposed by a local language model.

## Control flow

```text
SUMO traffic simulation
→ TraCI telemetry extraction
→ structured traffic-state prompt
→ local LLaMA 3.1 inference via Ollama
→ JSON signal-control decision
→ TraCI applies traffic-light action
→ metrics logging and dashboard visualisation
```

## SUMO simulation layer

SUMO models a single four-way intersection, vehicle routes and traffic-light phases. Each controller runs the configured scenario for 600 one-second simulation steps. The experiment is a software simulation; it does not connect to physical signal infrastructure.

## TraCI control interface

The Python controllers launch SUMO and use TraCI as both the observation and action interface. At each step, TraCI exposes the active signal phase, signal state, lane-level halting counts, vehicle waiting times and the number of vehicles currently in the network. The controller can then set a phase or its remaining duration.

## LLM decision layer

Every 10 simulation steps, the LLM controller aggregates lane queues into north, south, east and west approaches. It builds a structured prompt containing:

- current phase and elapsed phase time
- directional and corridor queue lengths
- current average waiting time
- bounded decision rules and the expected JSON schema

The prompt is sent to `llama3.1:8b` through the local Ollama generate endpoint. The expected response contains an `action` (`keep` or `switch`) and a `duration`. The parser extracts JSON, validates the action and clamps duration to 15–45 seconds. Connection failures, timeouts and unparseable responses use a conservative `keep` fallback.

## Metrics logging

Both controllers collect the same per-step traffic measurements and write them to separate CSV files in `Data/`. These include average waiting time, total and maximum queue length, active phase, vehicle count and per-lane queues. The LLM CSV also records the action and requested duration on inference steps.

The runners count departed and arrived vehicles in memory and print those totals at the end. Those throughput totals are not yet stored in the CSV files, which limits complete reproduction of the reported vehicles-processed comparison from saved artifacts alone.

## Streamlit dashboard

The dashboard loads both CSV files and visualises waiting time, queue length, peak vehicle count and the distribution of keep/switch decisions. It is a post-run comparison interface rather than part of the control loop.

## Baseline controller versus LLM controller

| Concern | Fixed-cycle baseline | LLM controller |
|---|---|---|
| Policy | Repeats a deterministic 70-second cycle | Queries the local model every 10 steps |
| Observation use | Does not adapt to telemetry | Includes current queues, waiting time and phase state in the prompt |
| Action | Enforces the phase implied by elapsed cycle time | Keeps or switches phase and sets a bounded duration |
| Failure mode | No model dependency | Falls back to `keep` when the model request or response fails |
| Output | Per-step traffic metrics | Per-step traffic metrics plus model decisions |

The baseline provides a simple reference policy. It is not an adaptive traffic-control baseline, so a stronger evaluation should also compare FlowLLM with a deterministic queue-responsive controller.
