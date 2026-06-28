# Stereotype Schema Extensions

## Problema

Parametri steriotipo sono liste piatte chiave-valore. Non esprimono:
- Dipendenze tra parametri (es. padding implicito da kernel_size)
- Scelta runtime dell'implementazione (es. kernel fp32 vs TensorRT)
- Validazione custom
- Logica di istantiazione (es. scaricare pesi pre-trained)

## Schema esteso proposto

```json
{
  "pythonClassName": "my_ops.AutoTunedConv2d",
  "category": "Layer",
  "view": { "color": "#4A90D9", "size": { "width": 120, "height": 60 } },
  "params": [
    {
      "name": "in_channels",
      "type": "int",
      "default": 64
    },
    {
      "name": "kernel_size",
      "type": "int",
      "default": 3,
      "description": "Dimensione kernel convoluzione"
    },
    {
      "name": "kernel_choice",
      "type": "enum",
      "default": "auto",
      "options": ["auto", "fp32", "fp16", "cudnn", "custom_trt"],
      "runtime_hint": "kernel_selection"
    },
    {
      "name": "autotune",
      "type": "bool",
      "default": true,
      "show_if": { "kernel_choice": "auto" }
    }
  ],
  "hooks_module": "my_ops.hooks",
  "hooks": {
    "on_param_change": "conv2d_on_config_change",
    "validate": "conv2d_validate_params",
    "on_instantiate": "conv2d_load_pretrained"
  }
}
```

## Cosa aggiunge

### 1. `runtime_hint`

Marca parametri che non vanno passati al costruttore Python ma usati per decisioni di compilazione/istantiazione.

Valori possibili:
- `kernel_selection` — sceglie implementazione kernel (fp32/fp16/trt)
- `compile_mode` — hint per torch.compile (reduce-overhead / max-autotune)
- `device_map` — hint per placement su multi-GPU
- `precision` — hint per AMP/dtype del modulo

### 2. `show_if`

Dipendenze visive tra parametri per il Sidebar. Un campo nascosto quando la condizione non è soddisfatta. La condizione è una mappa `{param_name: value}`.
- Valutata lato front-end in Svelte (reattività)
- Supporto AND (più coppie nella mappa)
- Supporto OR futuristico: array di condizioni

### 3. `type: "enum"` con `options`

Dropdown nativo nell'editor visuale. `SDropdown.svelte` già esiste — serve solo legarlo allo schema.

### 4. `hooks` + `hooks_module`

Callback Python eseguibili a compile-time (non runtime):

| Hook | Quando | Scopo |
|---|---|---|
| `on_param_change` | Utente modifica parametro nell'editor | Ricalcola parametri derivati (padding, stride) |
| `validate` | Prima di convert.py o clic "Validate" | Validazione custom (es. canali divisibili per head) |
| `on_instantiate` | Hydra istanzia il modulo | Setup post-creazione (carica pesi, inizializza buffer) |

Le hooks sono funzioni Python definite in `hooks_module` e chiamate da `convert.py` a compile-time.
Per feedback live nell'editor, una mappa JS di default copre i casi comuni (padding = kernel//2, ecc.).

### 5. `description` su ogni parametro

Tooltip informativo nel Sidebar. Banale ma manca.

## Impatto sui componenti

### convert.py

Aggiungere `resolve_runtime_hints(layer_dict, stereotype_meta)`:

```python
def resolve_runtime_hints(layer_dict, stereotype_meta):
    """Processa parametri marcati runtime_hint prima di Hydra instantiate."""
    for p in stereotype_meta.get("params", []):
        if "runtime_hint" in p and p["name"] in layer_dict:
            val = layer_dict.pop(p["name"])
            hint = p["runtime_hint"]
            if hint == "kernel_selection":
                _apply_kernel_selection(layer_dict, val, stereotype_meta)
            elif hint == "precision":
                _apply_precision_hint(layer_dict, val)
    return layer_dict
```

### Sidebar.svelte

- Leggere `type: "enum"` → renderizzare SDropdown con le options
- Leggere `show_if` → `$derived` per visibilità condizionale
- Leggere `description` → tooltip al hover

### convert.py: hooks

Se `hooks_module` è presente, importarlo a compile-time:

```python
import importlib
if "hooks_module" in stereo_meta:
    mod = importlib.import_module(stereo_meta["hooks_module"])
    if "validate" in stereo_meta.get("hooks", {}):
        fn = getattr(mod, stereo_meta["hooks"]["validate"])
        fn(layer_dict)
```

### Schema JSON: retrocompatibilità

- `params` array esiste già. Nuovi campi (`runtime_hint`, `show_if`, `options`) sono opzionali — steriotipi vecchi continuano a funzionare.
- `enum` type non riconosciuto da front-end vecchio → caduta a text input. Non rompe.
- `hooks_module` ignorato se assente.

## Priorità

Tre fasi:

1. **Subito:** `description` + `type: "enum"` con `options`. Sono i meno invasivi e danno più valore subito all'UX.
2. **Prossima:** `show_if` — pulisce il Sidebar da parametri irrilevanti quando certe opzioni sono selezionate.
3. **Futura:** `runtime_hint` + `hooks` — servono per utenti avanzati che vogliono kernel custom ottimizzati. Richiede modifica a convert.py e testing del meccanismo di hook a compile-time.
