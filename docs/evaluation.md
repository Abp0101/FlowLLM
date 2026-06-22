# Evaluation

## Current metrics

FlowLLM currently reports three main outcome measures:

- **Average waiting time:** the mean of the per-step average waiting time for vehicles present in the network.
- **Average queue length:** the mean total number of halting vehicles across controlled lanes per simulation step.
- **Vehicles processed:** run-level throughput reported by the controller's departed and arrived vehicle counters.

The saved CSV files also contain maximum queue length, active phase, current vehicle count and per-lane queue measurements. The LLM run records keep/switch decisions and durations on model-query steps.

## Current comparison

The repository compares a fixed-cycle baseline with the LLM controller over a 600-second single-intersection simulation. The included CSV files produce the reported approximate averages:

| Metric | Fixed-cycle baseline | LLM controller | Reported difference |
|---|---:|---:|---:|
| Average waiting time | 14.4 s | 12.4 s | 14% lower |
| Average queue length | 91.6 vehicles | 69.5 vehicles | 24% lower |
| Vehicles processed | 278 | 288 | 10 more |

Waiting time and queue length can be recalculated from the saved per-step CSVs. The throughput totals are currently printed at the end of a run but are not saved, so the repository should persist them before describing all benchmark outputs as fully reproducible.

## Known limitation

The current numbers appear to come from a limited prototype benchmark rather than repeated trials across controlled random seeds. They show that the current LLM run outperformed the fixed-cycle run on the reported scenario, but they do not establish statistical significance, robustness across traffic patterns or suitability for real-world control.

## Recommended next evaluation steps

1. Run each controller over 20–50 matched random seeds.
2. Report the mean and standard deviation for every outcome metric.
3. Add confidence intervals for the difference between controllers.
4. Compare against a rule-based adaptive controller, not only a fixed cycle.
5. Log invalid or repaired model outputs separately from valid outputs.
6. Log end-to-end model latency for every inference request.
7. Measure fallback-controller usage, including timeout and parse-failure reasons.
8. Record decision traces containing the observed state, raw model response, validated action and resulting traffic state.

## Reproducible benchmark protocol

For each trial, record the SUMO seed, route configuration, controller version, model name, model parameters and prompt version. Run baseline and candidate controllers against matched demand. Persist a run summary alongside per-step data, and compute aggregate statistics from the saved artifacts rather than manually copied console output.
