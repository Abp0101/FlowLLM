# Roadmap

## Evaluation reliability

- [ ] Build a multi-seed benchmark runner for matched baseline and LLM trials.
- [ ] Report mean, standard deviation and confidence intervals.
- [ ] Add a queue-responsive adaptive non-LLM baseline.
- [ ] Persist run-level throughput and configuration metadata with each benchmark.

## Controller reliability

- [ ] Expand invalid-output handling with explicit validation outcomes and reason codes.
- [ ] Add a timeout fallback to the fixed-cycle controller.
- [ ] Record decision traces: observed state, prompt version, raw response, validated action, latency and outcome.
- [ ] Add a bounded self-critique or self-correction loop for invalid or low-confidence decisions.

## Experimental scope

- [ ] Extend the environment to a multi-intersection grid simulation.
- [ ] Run experiments using real-world open traffic datasets as simulation demand inputs.

Each milestone should include a reproducible evaluation before its results are promoted in the README. Real-world data experiments should remain simulation studies unless separate deployment and safety work is completed.
