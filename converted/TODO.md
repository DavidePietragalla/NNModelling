# DAG Parallel Forward Execution

## Problema

Forward di `Net` e `Subflow` esegue nodi con Kahn's topological sort in singolo loop FIFO (`pop(0)`). Fork-join topologie (es. due Conv2d in parallelo con join Addition) eseguono i rami seriatamente, uno dopo l'altro, senza sfruttare parallelismo GPU.

## Tre approcci considerati

### 1. CUDA Streams multipli manuali

Quando Kahn's produce N nodi ready (in-degree=0), lanciarli su N stream CUDA separati invece che sul default stream. Poi sync event alla join.

```
# Invece di:
for node in ready_nodes:
    out = compute(node)  # FIFO su default stream

# Fare:
for i, node in enumerate(ready_nodes):
    with torch.cuda.stream(streams[i]):
        out = compute(node)
# sync: torch.cuda.synchronize() o eventi per join specifici
```

**Pro:**
- Massimo controllo: ogni kernel lanciato su stream specifico
- GPU può eseguire kernel da rami indipendenti concorrentemente (es. due Conv2d in parallelo)
- Nessuna dipendenza da compilatore

**Contro:**
- Overhead context switching GPU — non sempre conviene (rametti leggeri peggiorano)
- Fragile: bisogna gestire event, sync, decidere quando streamare
- Non portabile su CPU
- Integrazione delicata con Lightning (salvataggio/checkpoint)

### 2. torch.compile

`torch.compile(model, backend="inductor")` analizza grafo FX e può parallelizzare automaticamente rami indipendenti.

**Pro:**
- Zero modifiche al codice — decorate e via
- Ottimizzazioni gratis (fusion, kernel autotune)
- Standard PyTorch 2.x, mantenuto da Meta

**Contro:**
- **Non applicabile nel nostro caso.** Forward loop `while queue: pop(0), if type dispatch, dict lookup` è control flow Python dinamico. `torch.compile` traccia solo operazioni tensoriali. Non può unrollare un loop che dipende dalla topologia del modello (diversa per ogni diagramma).
- Non funziona se la struttura del grafo cambia tra forward. Il nostro `module_dict` è fixo ma il traversal loop è dinamico — compile non vede i tensor ops dentro la queue logic.
- Quando funziona: reti lineari (Sequential puro), non DAG dinamici.

### 3. Wavefront Scheduling (scelto)

Analisi statica del DAG fatta una volta in `__init__`, produce gruppi di nodi parallelizzabili (wavefronts). Forward esegue wavefront su stream pre-alllocati.

**Inizializzazione:** compute wavefronts da DAG:
```
A → B, A → C, B → D, C → D

in-degree init: A=0, B=1, C=1, D=2

Wavefront 0: [A]          (in-degree=0)
Wavefront 1: [B, C]       (A consumato, B e C → in-degree=0)
Wavefront 2: [D]           (B e C consumati, D → in-degree=0)
```

**Forward loop:**
```python
def forward(self, x):
    node_inputs = {self.root_id: x}
    for wf in self.wavefronts:
        futures = {}
        for i, node_id in enumerate(wf):
            s = self.streams[i]
            with torch.cuda.stream(s):
                node_inputs[node_id] = compute(node_id, node_inputs)
        # sync tra wavefront: ogni ciclo forward finito prima del prossimo
    return node_inputs[self.output_id]
```

Stream multipli solo dentro ogni wavefront. Sync garantita tra wavefront successivi.

**Pro:**
- Analisi DAG una volta (init), forward è hot-path lineare
- GPU parallelism esplicito: rami fork eseguiti concorrentemente
- Wavefront di ampiezza 1 = nessun overhead stream (caso sequenziale puro)
- Compatibile con Hydra: DAG noto dopo instantiazione, wavefront computabile
- Si integra con net/base.py e ops/subflow.py entrambi

**Contro:**
- Wavefront statico: se runtime cambia topologia, non adatta (non è il nostro caso)
- Stream context switching overhead (trascurabile per nodi pesanti come Conv2d/Linear)
- Richiede GPU (CPU fallback a loop sequenziale)
- Non banale da testare (confronto tensor matching seriale vs parallelo)

## Priorità: MEDIA

Wavefront scheduling dà speedup reale solo su modelli con fork multipli pesanti (es. due rami Conv2d profondi). Per transformer encoder con attenzione `vmap`tizzata già efficiently, bottleneck non è il loop scheduling ma i kernel attention matmul. Lo scheduler parallelo aiuta di più su architetture multi-branch (Inception, ResNeXt, NASNet).

## Riferimenti

- **CUDA Streams:** [PyTorch CUDA Semantics — torch.cuda.Stream](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- **torch.compile:** [PyTorch 2.x torch.compile docs](https://pytorch.org/docs/stable/generated/torch.compile.html)
- **torch.func vmap:** [PyTorch vmap docs](https://pytorch.org/docs/stable/generated/torch.func.vmap.html) (già usato in HorizontalRepeat)
- **Wavefront scheduling:** [Kahn-Merrill algorithm per DAG parallel scheduling](https://en.wikipedia.org/wiki/Topological_sorting) — concetto base, adattato per GPU streams
- **DAG parallel execution on GPU:** [Catanzaro et al. "Efficient Topological Sort for GPU"](https://www.sciencedirect.com/science/article/pii/S1877050915037350) (paper)
- **Parallel wavefront in ML compilers:** [XLA: Optimized Compiler for Machine Learning](https://www.tensorflow.org/xla) — XLA usa scheduling a wavefront per SPMD parallelism
