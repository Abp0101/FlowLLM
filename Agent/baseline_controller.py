"""
FlowLLM — Baseline Fixed-Cycle Traffic Light Controller
========================================================
Connects to SUMO via TraCI, runs a fixed-cycle signal plan
(31s green / 4s yellow, alternating NS ↔ EW), and logs
per-step traffic metrics to Data/baseline_metrics.csv.

Usage:
    python Agent/baseline_controller.py          # headless (sumo)
    python Agent/baseline_controller.py --gui    # with SUMO GUI
"""

import os
import sys
import csv
import argparse

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
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "Data", "baseline_metrics.csv")

# ---------------------------------------------------------------------------
# Traffic light phase definitions
# ---------------------------------------------------------------------------
# Phase index mapping (matches intersection.tll.xml):
#   0 = NS green  (31 s)
#   1 = NS yellow ( 4 s)
#   2 = EW green  (31 s)
#   3 = EW yellow ( 4 s)
PHASE_DURATIONS = [31, 4, 31, 4]        # seconds per phase
TL_ID           = "center"              # traffic light ID in the network
TOTAL_PHASES    = len(PHASE_DURATIONS)
SIM_END         = 600                   # total simulation time in seconds


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FlowLLM Baseline Controller")
    parser.add_argument("--gui", action="store_true",
                        help="Launch SUMO with the graphical interface")
    return parser.parse_args()


def start_sumo(use_gui: bool):
    """
    Launch SUMO (or sumo-gui) as a subprocess and connect via TraCI.
    The simulation is configured by Simulation/run.sumocfg.
    """
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CFG,
        "--step-length", "1",          # 1-second simulation steps
        "--no-step-log", "true",       # suppress per-step console output
        "--waiting-time-memory", "600" # track waiting time for full sim
    ]
    traci.start(sumo_cmd)
    print(f"[FlowLLM] SUMO started ({'GUI' if use_gui else 'headless'}) "
          f"with config: {SUMO_CFG}")


def collect_metrics(step: int) -> dict:
    """
    Collect traffic metrics for the current simulation step.

    Returns a dict with:
        - sim_time:           current simulation second
        - phase_index:        active traffic light phase (0–3)
        - phase_state:        signal state string (e.g. 'GGGgGGGgrrrrrrrr')
        - total_queue_length: sum of halting vehicles across all lanes
        - avg_waiting_time:   mean waiting time of all vehicles in the network
        - max_queue_length:   worst single-lane queue length
        - vehicle_count:      number of vehicles currently in the network
        - per-lane columns:   queue_<lane_id> for every lane at the junction
    """
    metrics = {
        "sim_time": step,
        "phase_index": traci.trafficlight.getPhase(TL_ID),
        "phase_state": traci.trafficlight.getRedYellowGreenState(TL_ID),
    }

    # --- Queue lengths per lane -------------------------------------------
    # Get all lanes controlled by the traffic light's incoming edges
    controlled_lanes = traci.trafficlight.getControlledLanes(TL_ID)
    # Remove duplicates while preserving order
    seen = set()
    unique_lanes = [l for l in controlled_lanes if not (l in seen or seen.add(l))]

    lane_queues = {}
    for lane_id in unique_lanes:
        # getLastStepHaltingNumber = vehicles with speed < 0.1 m/s on this lane
        queue = traci.lane.getLastStepHaltingNumber(lane_id)
        lane_queues[lane_id] = queue

    # Aggregate queue metrics
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

    # Add individual lane queue columns for detailed analysis
    for lane_id, queue in lane_queues.items():
        metrics[f"queue_{lane_id}"] = queue

    return metrics


def enforce_fixed_cycle(step: int):
    """
    Enforce the fixed-cycle signal plan by manually setting the phase
    based on the elapsed simulation time. This ensures the controller
    runs exactly the timing we specify, independent of SUMO's internal
    traffic light logic.

    Cycle: NS green (31s) → NS yellow (4s) → EW green (31s) → EW yellow (4s)
    Total cycle length = 70 seconds
    """
    cycle_length = sum(PHASE_DURATIONS)               # 70 seconds
    time_in_cycle = step % cycle_length                # 0–69

    # Determine which phase we're in
    elapsed = 0
    target_phase = 0
    for i, duration in enumerate(PHASE_DURATIONS):
        elapsed += duration
        if time_in_cycle < elapsed:
            target_phase = i
            break

    # Only send a TraCI command if the phase needs to change
    current_phase = traci.trafficlight.getPhase(TL_ID)
    if current_phase != target_phase:
        traci.trafficlight.setPhase(TL_ID, target_phase)


def run_simulation(use_gui: bool):
    """
    Main simulation loop:
      1. Start SUMO
      2. For each timestep, enforce the fixed cycle and collect metrics
      3. Save metrics to CSV
      4. Print summary statistics
    """
    start_sumo(use_gui)

    all_metrics = []          # accumulates one dict per simulation step
    total_departed = 0        # running count of all vehicles that entered
    total_arrived  = 0        # running count of all vehicles that finished

    print(f"[FlowLLM] Running baseline simulation for {SIM_END} seconds...")
    print(f"[FlowLLM] Fixed cycle: {PHASE_DURATIONS} (total {sum(PHASE_DURATIONS)}s)")

    # ------------------------------------------------------------------
    # Main simulation loop — one iteration per simulation second
    # ------------------------------------------------------------------
    step = 0
    while step < SIM_END:
        # Advance SUMO by one step
        traci.simulationStep()

        # Enforce our fixed-cycle signal timing
        enforce_fixed_cycle(step)

        # Collect traffic metrics for this timestep
        step_metrics = collect_metrics(step)
        all_metrics.append(step_metrics)

        # Track total throughput
        total_departed += traci.simulation.getDepartedNumber()
        total_arrived  += traci.simulation.getArrivedNumber()

        # Progress indicator every 100 seconds
        if step % 100 == 0 and step > 0:
            print(f"  [t={step:>4d}s]  vehicles in network: "
                  f"{step_metrics['vehicle_count']:>3d}  |  "
                  f"queue: {step_metrics['total_queue_length']:>3d}  |  "
                  f"avg wait: {step_metrics['avg_waiting_time']:.1f}s")

        step += 1

    # ------------------------------------------------------------------
    # Simulation complete — close TraCI connection
    # ------------------------------------------------------------------
    traci.close()
    print(f"\n[FlowLLM] Simulation complete.")

    # ------------------------------------------------------------------
    # Save metrics to CSV using pandas
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[FlowLLM] Metrics saved to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  BASELINE SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Simulation duration:     {SIM_END} seconds")
    print(f"  Cycle length:            {sum(PHASE_DURATIONS)} seconds")
    print(f"  Total vehicles departed: {total_departed}")
    print(f"  Total vehicles arrived:  {total_arrived}")
    print(f"  Avg waiting time:        "
          f"{df['avg_waiting_time'].mean():.2f} seconds")
    print(f"  Max queue length:        "
          f"{df['max_queue_length'].max()} vehicles")
    print(f"  Avg queue length:        "
          f"{df['total_queue_length'].mean():.1f} vehicles")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_simulation(use_gui=args.gui)
