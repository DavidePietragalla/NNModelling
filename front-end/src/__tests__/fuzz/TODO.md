# Fuzz Testing TODO

- [ ] **Fuzzer #2 — Forward Pass** (NNTree → Python)
  Genera NNTree JSON con shape compatibili, lancia `convert.py` + `Net.forward()`,
  verifica output finito (no NaN/Inf) e gradienti presenti.
  Richiede: shape propagator, subprocess Python orchestration (vedi test di integrazione).
