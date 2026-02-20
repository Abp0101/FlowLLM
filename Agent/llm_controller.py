"""
FlowLLM — LLM-Driven Traffic Light Controller
===============================================
Connects to SUMO via TraCI and queries a local Ollama LLM
(llama3.1:8b) every 10 simulation steps to decide whether
to keep or switch the traffic light phase.

The LLM receives the current intersection state (queue lengths,
waiting times, phase info) and responds with a JSON decision:
  {"action": "keep" or "switch", "duration": 15-45}

Usage:
    python Agent/llm_controller.py          # headless (sumo)
    python Agent/llm_controller.py --gui    # with SUMO GUI
"""

import os
import sys
import json
import re
import argparse
import requests

# ---------------------------------------------------------------------------
# TraCI import — uses SUMO_HOME to locate the traci package
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("ERROR: SUMO_HOME environment variable is not set. "
             "Please set it to your SUMO installation directory.")

import traci  # noqa: E402  (must come after path setup)
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Project paths (resolved relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SUMO_CFG     = os.path.join(PROJECT_ROOT, "Simulation", "run.sumocfg")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "Data", "llm_metrics.csv")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
TL_ID             = "center"                            # traffic light id
SIM_END           = 600                                 # simulation length (s)
LLM_QUERY_INTERVAL = 10                                # query LLM every N steps
OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "llama3.1:8b"

# Phase mapping (matches intersection.tll.xml)
PHASE_NAMES = {
    0: "NS_GREEN",
    1: "NS_YELLOW",
    2: "EW_GREEN",
    3: "EW_YELLOW",
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FlowLLM LLM Controller")
    parser.add_argument("--gui", action="store_true",
                        help="Launch SUMO with the graphical interface")
    return parser.parse_args()


def start_sumo(use_gui: bool):
    """
    Launch SUMO (or sumo-gui) as a subprocess and connect via TraCI.
    """
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CFG,
        "--step-length", "1",
        "--no-step-log", "true",
        "--waiting-time-memory", "600",
    ]
    traci.start(sumo_cmd)
    print(f"[FlowLLM-LLM] SUMO started ({'GUI' if use_gui else 'headless'}) "
          f"with config: {SUMO_CFG}")


# ---------------------------------------------------------------------------
# Metrics collection (identical to baseline for fair comparison)
# ---------------------------------------------------------------------------
def collect_metrics(step: int) -> dict:
    """
    Collect traffic metrics for the current simulation step.

    Returns a dict with:
        - sim_time, phase_index, phase_state
        - total / max / per-lane queue lengths
        - avg_waiting_time, vehicle_count
        - llm_action: what the LLM decided (if applicable this step)
        - llm_duration: duration the LLM requested
    """
    metrics = {
        "sim_time": step,
        "phase_index": traci.trafficlight.getPhase(TL_ID),
        "phase_state": traci.trafficlight.getRedYellowGreenState(TL_ID),
    }

    # --- Queue lengths per lane -------------------------------------------
    controlled_lanes = traci.trafficlight.getControlledLanes(TL_ID)
    seen = set()
    unique_lanes = [l for l in controlled_lanes if not (l in seen or seen.add(l))]

    lane_queues = {}
    for lane_id in unique_lanes:
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        lane_queues[lane_id] = queue

    queue_values = list(lane_queues.values())
    metrics["total_queue_length"] = sum(queue_values) if queue_values else 0
    metrics["max_queue_length"]   = max(queue_values) if queue_values else 0

    # --- Waiting time per vehicle -----------------------------------------
    vehicle_ids = traci.vehicle.getIDList()
    metrics["vehicle_count"] = len(vehicle_ids)

    if vehicle_ids:
        total_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)
        metrics["avg_waiting_time"] = round(total_wait / len(vehicle_ids), 2)
    else:
        metrics["avg_waiting_time"] = 0.0

    # Per-lane queue columns
    for lane_id, queue in lane_queues.items():
        metrics[f"queue_{lane_id}"] = queue

    return metrics, lane_queues


# ---------------------------------------------------------------------------
# Intersection state builder (sent to the LLM)
# ---------------------------------------------------------------------------
def build_intersection_state(step: int, lane_queues: dict, avg_wait: float,
                             phase_duration_elapsed: int) -> dict:
    """
    Build a compact state summary of the intersection for the LLM prompt.

    Groups lanes by direction (north, south, east, west) and reports
    the total queue per direction plus the current phase, how long it
    has been active, and the average waiting time.
    """
    # Group lane queues by direction (edge name prefix)
    direction_queues = {"north": 0, "south": 0, "east": 0, "west": 0}
    for lane_id, queue in lane_queues.items():
        # Lane IDs look like: "north_to_center_0", "east_to_center_1"
        for direction in direction_queues:
            if lane_id.startswith(direction):
                direction_queues[direction] += queue
                break

    current_phase = traci.trafficlight.getPhase(TL_ID)

    state = {
        "simulation_time": step,
        "current_phase": PHASE_NAMES.get(current_phase, f"UNKNOWN_{current_phase}"),
        "phase_index": current_phase,
        "queue_lengths": direction_queues,
        "average_waiting_time": avg_wait,
        "phase_active_seconds": phase_duration_elapsed,
    }
    return state


# ---------------------------------------------------------------------------
# LLM interaction via Ollama
# ---------------------------------------------------------------------------
def build_llm_prompt(state: dict) -> str:
    """
    Build a detailed prompt that provides the LLM with full intersection
    context, clear queue data, phase timing, and explicit prioritisation
    instructions. Includes an example of the expected JSON response.
    """
    q = state["queue_lengths"]
    ns_total = q["north"] + q["south"]
    ew_total = q["east"] + q["west"]

    # Identify the most congested direction for the hint
    busiest = max(q, key=q.get)

    prompt = (
        # --- System context ------------------------------------------------
        "You are an expert traffic signal controller optimising a single "
        "4-way intersection. Your goal is to MINIMISE average vehicle "
        "waiting time and prevent any single direction from building up "
        "excessive queues.\n\n"

        # --- Current intersection state -----------------------------------
        "=== CURRENT INTERSECTION STATE ===\n"
        f"Simulation time: {state['simulation_time']}s / 600s\n"
        f"Current phase:   {state['current_phase']}\n"
        f"Phase active for: {state['phase_active_seconds']}s\n"
        f"Avg waiting time: {state['average_waiting_time']:.1f}s\n\n"

        # --- Queue lengths clearly labelled --------------------------------
        "=== QUEUE LENGTHS (vehicles waiting) ===\n"
        f"  North approach: {q['north']} vehicles\n"
        f"  South approach: {q['south']} vehicles\n"
        f"  --> N-S corridor total: {ns_total}\n"
        f"  East approach:  {q['east']} vehicles\n"
        f"  West approach:  {q['west']} vehicles\n"
        f"  --> E-W corridor total: {ew_total}\n"
        f"  Busiest direction: {busiest.upper()} ({q[busiest]} vehicles)\n\n"

        # --- Decision rules ------------------------------------------------
        "=== DECISION RULES ===\n"
        "1. If the CURRENT phase is GREEN for the corridor with the LONGEST "
        "queues, KEEP it to let those vehicles clear.\n"
        "2. If the OPPOSITE corridor has significantly more queued vehicles, "
        "SWITCH to serve them.\n"
        "3. Do NOT switch if the phase has been active for less than 15s "
        "(too short causes inefficiency).\n"
        "4. Do NOT keep a phase for more than 45s (starves the other "
        "corridor).\n"
        "5. Set duration based on queue size: larger queues need more "
        "green time.\n\n"

        # --- Response format -----------------------------------------------
        "=== RESPOND WITH ONLY THIS JSON (no explanation) ===\n"
        '{"action": "keep" or "switch", "duration": <15-45>}\n\n'

        # --- Example -------------------------------------------------------
        "Example: if NS is green and North has 12, South has 8, East has 3, "
        "West has 2, you should keep NS green:\n"
        '{"action": "keep", "duration": 30}\n'
    )
    return prompt


def query_ollama(prompt: str) -> dict:
    """
    Send a prompt to the local Ollama instance and parse the JSON response.

    Returns a dict with keys 'action' ('keep' or 'switch') and 'duration'.
    Falls back to {"action": "keep", "duration": 10} on any failure.
    """
    fallback = {"action": "keep", "duration": 10}

    try:
        # Send request to Ollama API (non-streaming for simpler parsing)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,     # low temp for more deterministic output
                    "num_predict": 50,      # limit response length
                },
            },
            timeout=30,  # 30s timeout to avoid hanging
        )
        response.raise_for_status()

        # Extract the generated text from Ollama's response
        raw_text = response.json().get("response", "")

        # Parse JSON from the LLM response (may contain extra text around it)
        decision = parse_llm_response(raw_text)
        return decision

    except requests.exceptions.ConnectionError:
        print("  [LLM] WARNING: Cannot connect to Ollama. Is it running? "
              "Falling back to 'keep'.")
        return fallback
    except requests.exceptions.Timeout:
        print("  [LLM] WARNING: Ollama request timed out. Falling back to 'keep'.")
        return fallback
    except Exception as e:
        print(f"  [LLM] WARNING: Unexpected error: {e}. Falling back to 'keep'.")
        return fallback


def parse_llm_response(raw_text: str) -> dict:
    """
    Extract and validate the JSON decision from the LLM's raw text output.

    Handles cases where the LLM wraps JSON in markdown code blocks or
    includes extra explanation text around the JSON object.

    Returns validated dict with 'action' and 'duration' keys.
    """
    fallback = {"action": "keep", "duration": 10}

    # Strip markdown code fences if present (```json ... ```)
    cleaned = raw_text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)

    # Try to find a JSON object in the response
    json_match = re.search(r"\{[^}]+\}", cleaned)
    if not json_match:
        print(f"  [LLM] WARNING: No JSON found in response: {raw_text[:100]}")
        return fallback

    try:
        decision = json.loads(json_match.group())
    except json.JSONDecodeError:
        print(f"  [LLM] WARNING: Invalid JSON: {json_match.group()}")
        return fallback

    # --- Validate and sanitise the decision --------------------------------
    action = str(decision.get("action", "keep")).lower().strip()
    if action not in ("keep", "switch"):
        print(f"  [LLM] WARNING: Unknown action '{action}', defaulting to 'keep'.")
        action = "keep"

    try:
        duration = int(decision.get("duration", 20))
    except (ValueError, TypeError):
        duration = 20

    # Clamp duration to the allowed range [15, 45]
    duration = max(15, min(45, duration))

    return {"action": action, "duration": duration}


# ---------------------------------------------------------------------------
# Traffic light control logic
# ---------------------------------------------------------------------------
def apply_llm_decision(decision: dict, current_phase: int):
    """
    Apply the LLM's decision to the traffic light.

    - "keep":   stay on the current green phase for 'duration' seconds
    - "switch": transition through yellow, then switch to the opposite
                green phase and hold for 'duration' seconds

    Yellow phases (1 and 3) are always 4 seconds and happen automatically.
    """
    action   = decision["action"]
    duration = decision["duration"]

    if action == "switch":
        # If currently on a green phase, advance to the yellow phase
        # Phase cycle: 0(NS green) → 1(NS yellow) → 2(EW green) → 3(EW yellow)
        if current_phase == 0:
            # NS green → NS yellow → EW green
            traci.trafficlight.setPhase(TL_ID, 1)              # go to yellow
            traci.trafficlight.setPhaseDuration(TL_ID, 4)      # yellow lasts 4s
        elif current_phase == 2:
            # EW green → EW yellow → NS green
            traci.trafficlight.setPhase(TL_ID, 3)              # go to yellow
            traci.trafficlight.setPhaseDuration(TL_ID, 4)      # yellow lasts 4s
        else:
            # Already in a yellow phase — let it finish naturally
            pass

        print(f"  [LLM] -> SWITCH (hold next green for {duration}s)")
    else:
        # Keep the current phase and set its remaining duration
        if current_phase in (0, 2):  # only set duration on green phases
            traci.trafficlight.setPhaseDuration(TL_ID, duration)
        print(f"  [LLM] -> KEEP current phase for {duration}s")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_simulation(use_gui: bool):
    """
    Main simulation loop:
      1. Start SUMO
      2. Every step: collect metrics
      3. Every LLM_QUERY_INTERVAL steps: query LLM and apply decision
      4. Save all metrics to CSV
      5. Print summary
    """
    start_sumo(use_gui)

    all_metrics    = []       # one dict per simulation step
    llm_decisions  = []       # log of all LLM interactions
    total_departed = 0
    total_arrived  = 0

    # Track how long the current phase has been active
    last_phase      = -1      # previous phase index (init to impossible value)
    phase_start_step = 0      # step when the current phase started

    print(f"[FlowLLM-LLM] Running LLM-controlled simulation for {SIM_END}s")
    print(f"[FlowLLM-LLM] LLM query interval: every {LLM_QUERY_INTERVAL} steps")
    print(f"[FlowLLM-LLM] Model: {OLLAMA_MODEL} @ {OLLAMA_URL}")

    # ------------------------------------------------------------------
    # Main loop — one iteration per simulation second
    # ------------------------------------------------------------------
    step = 0
    while step < SIM_END:
        # Advance SUMO by one step
        traci.simulationStep()

        # Collect traffic metrics for this step
        step_metrics, lane_queues = collect_metrics(step)

        # Track throughput
        total_departed += traci.simulation.getDepartedNumber()
        total_arrived  += traci.simulation.getArrivedNumber()

        # Track phase duration for the LLM prompt
        current_phase_now = traci.trafficlight.getPhase(TL_ID)
        if current_phase_now != last_phase:
            phase_start_step = step
            last_phase = current_phase_now
        phase_duration_elapsed = step - phase_start_step

        # ----------------------------------------------------------
        # Query the LLM every LLM_QUERY_INTERVAL steps
        # ----------------------------------------------------------
        if step > 0 and step % LLM_QUERY_INTERVAL == 0:
            # Build the intersection state for the LLM
            state = build_intersection_state(
                step, lane_queues, step_metrics["avg_waiting_time"],
                phase_duration_elapsed
            )

            # Build the prompt and query Ollama
            prompt = build_llm_prompt(state)
            print(f"\n  [t={step:>4d}s] Querying LLM... ", end="", flush=True)
            decision = query_ollama(prompt)

            # Apply the LLM's decision to the traffic light
            current_phase = traci.trafficlight.getPhase(TL_ID)
            apply_llm_decision(decision, current_phase)

            # Log the decision alongside the step metrics
            step_metrics["llm_action"]   = decision["action"]
            step_metrics["llm_duration"] = decision["duration"]

            # Store decision for the summary
            llm_decisions.append({
                "time": step,
                "phase_before": PHASE_NAMES.get(current_phase, str(current_phase)),
                **decision,
            })
        else:
            # No LLM query this step
            step_metrics["llm_action"]   = ""
            step_metrics["llm_duration"] = ""

        all_metrics.append(step_metrics)

        # Progress indicator every 100 seconds
        if step % 100 == 0 and step > 0 and step % LLM_QUERY_INTERVAL != 0:
            print(f"  [t={step:>4d}s]  vehicles: "
                  f"{step_metrics['vehicle_count']:>3d}  |  "
                  f"queue: {step_metrics['total_queue_length']:>3d}  |  "
                  f"avg wait: {step_metrics['avg_waiting_time']:.1f}s")

        step += 1

    # ------------------------------------------------------------------
    # Simulation complete
    # ------------------------------------------------------------------
    traci.close()
    print(f"\n[FlowLLM-LLM] Simulation complete.")

    # ------------------------------------------------------------------
    # Save metrics to CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[FlowLLM-LLM] Metrics saved to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Print summary (same format as baseline for easy comparison)
    # ------------------------------------------------------------------
    total_queries  = len(llm_decisions)
    switch_count   = sum(1 for d in llm_decisions if d["action"] == "switch")
    keep_count     = sum(1 for d in llm_decisions if d["action"] == "keep")

    print("\n" + "=" * 60)
    print("  LLM-CONTROLLED SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Simulation duration:     {SIM_END} seconds")
    print(f"  Total vehicles departed: {total_departed}")
    print(f"  Total vehicles arrived:  {total_arrived}")
    print(f"  Avg waiting time:        "
          f"{df['avg_waiting_time'].mean():.2f} seconds")
    print(f"  Max queue length:        "
          f"{df['max_queue_length'].max()} vehicles")
    print(f"  Avg queue length:        "
          f"{df['total_queue_length'].mean():.1f} vehicles")
    print(f"  ---")
    print(f"  LLM queries made:        {total_queries}")
    print(f"  LLM 'switch' decisions:  {switch_count}")
    print(f"  LLM 'keep' decisions:    {keep_count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_simulation(use_gui=args.gui)
