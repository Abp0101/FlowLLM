# Limitations

- **Prototype simulation:** FlowLLM is a software experiment, not a real-world traffic deployment.
- **Single-intersection environment:** the current network does not represent coordination, spillback or routing effects across multiple junctions.
- **Limited validation runs:** the reported comparison appears to use a limited prototype benchmark unless additional seeded runs are performed and documented.
- **Inference latency:** local LLM latency may affect real-time use. The stepped simulation does not demonstrate that deadlines could be met in a physical controller.
- **Output validation:** model outputs require validation before action. The current parser validates action names and duration bounds, but it does not provide a formal safety model for signal plans.
- **Fallback observability:** request and parse failures fall back to a keep decision, but fallback frequency and reason are not yet persisted as evaluation metrics.
- **No physical safety guarantee:** the project provides no guarantee of safe operation in physical traffic systems and has not been assessed against traffic-control safety standards.
- **Prototype results only:** the reported results should not be interpreted as production traffic-control performance.

The current experiment is useful for studying structured LLM decisions and feedback-driven evaluation in simulation. Broader claims require repeated benchmarks, stronger baselines, failure analysis and domain-specific safety validation.
