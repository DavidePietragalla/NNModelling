<script lang="ts">
  import { onMount } from "svelte";
  import type { Diagram } from "../Diagram.svelte";
  import { NNTree } from "../conversion/nnTree";
  import {
    cancelTrainingJob,
    listDatasets,
    listTrainingJobs,
    submitTrainingJob,
    trainingEventsUrl,
    type DatasetInfo,
    type DatasetParameter,
    type TrainingJobRequest,
    type TrainingJobStatus,
  } from "../training/api";

  interface Props {
    diagram: Diagram;
    onClose: () => void;
  }

  let { diagram, onClose }: Props = $props();

  let datasets = $state<DatasetInfo[]>([]);
  let jobs = $state<TrainingJobStatus[]>([]);
  let selectedDataset = $state("");
  let datasetParams = $state<Record<string, string>>({});
  let maxEpochs = $state("20");
  let learningRate = $state("0.001");
  let batchSize = $state("32");
  let numWorkers = $state("4");
  let trainSize = $state("0.8");
  let optimizerTarget = $state("torch.optim.Adam");
  let accelerator = $state("auto");
  let patience = $state("3");
  let minDelta = $state("0");
  let seed = $state("42");
  let wandbProject = $state("NeuralNetworks");
  let wandbMode = $state("online");
  let overridesText = $state("");
  let cpu = $state("4");
  let memoryGb = $state("8");
  let gpu = $state("0");
  let gpuMemoryGb = $state("");
  let gpuType = $state("");
  let node = $state("");
  let priority = $state("0");
  let selectedJobId = $state<string | null>(null);
  let loading = $state(false);
  let loadingJobs = $state(false);
  let errorMessage = $state("");
  let successMessage = $state("");
  let eventSource: EventSource | null = null;
  let refreshTimer: ReturnType<typeof setInterval> | undefined;

  let selectedDatasetInfo = $derived(
    datasets.find((dataset) => dataset.target === selectedDataset) ?? null,
  );

  onMount(() => {
    void loadDatasets();
    void refreshJobs();
    refreshTimer = setInterval(() => void refreshJobs(), 3000);
    return () => {
      if (refreshTimer) clearInterval(refreshTimer);
      eventSource?.close();
    };
  });

  async function loadDatasets() {
    try {
      datasets = await listDatasets();
      if (!selectedDataset && datasets.length > 0) selectDataset(datasets[0]);
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
    }
  }

  async function refreshJobs() {
    loadingJobs = true;
    try {
      jobs = await listTrainingJobs();
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
    } finally {
      loadingJobs = false;
    }
  }

  function selectDataset(dataset: DatasetInfo) {
    selectedDataset = dataset.target;
    datasetParams = Object.fromEntries(
      dataset.parameters.map((parameter) => [parameter.name, String(parameter.default ?? "")]),
    );
  }

  function setDatasetParameter(parameter: DatasetParameter, event: Event) {
    const value = (event.currentTarget as HTMLInputElement).value;
    datasetParams = { ...datasetParams, [parameter.name]: value };
  }

  function coerce(value: string, type: string): unknown {
    if (type === "int") return Number.parseInt(value, 10);
    if (type === "float") return Number.parseFloat(value);
    if (type === "bool") return value === "true";
    return value;
  }

  function buildRequest(): TrainingJobRequest {
    if (!selectedDatasetInfo) throw new Error("Seleziona un dataset");
    const datasetConfig: Record<string, unknown> = { _target_: selectedDataset };
    for (const parameter of selectedDatasetInfo.parameters) {
      const value = datasetParams[parameter.name];
      if (value !== undefined && value !== "") datasetConfig[parameter.name] = coerce(value, parameter.type);
    }
    const nntree = JSON.parse(new NNTree(diagram).toJson()) as Record<string, unknown>;
    return {
      schema_version: 1,
      network: { format: "nntree", value: nntree },
      training: {
        seed: Number.parseInt(seed, 10),
        dataset: {
          ...datasetConfig,
          batch_size: Number.parseInt(batchSize, 10),
          num_workers: Number.parseInt(numWorkers, 10),
          train_size: Number.parseFloat(trainSize),
        },
        optimizer: { _target_: optimizerTarget, lr: Number.parseFloat(learningRate) },
        trainer: { max_epochs: Number.parseInt(maxEpochs, 10), accelerator },
        wandb: { project: wandbProject, mode: wandbMode },
        early_stopping: {
          patience: Number.parseInt(patience, 10),
          min_delta: Number.parseFloat(minDelta),
        },
        overrides: overridesText.split("\n").map((line) => line.trim()).filter(Boolean),
      },
      resources: {
        cpu: Number.parseInt(cpu, 10),
        memory_gb: Number.parseFloat(memoryGb),
        gpu: Number.parseInt(gpu, 10),
        ...(gpuMemoryGb ? { gpu_memory_gb: Number.parseFloat(gpuMemoryGb) } : {}),
        ...(gpuType ? { gpu_type: gpuType } : {}),
        ...(node ? { node } : {}),
      },
      priority: Number.parseInt(priority, 10),
    };
  }

  async function submit() {
    loading = true;
    errorMessage = "";
    successMessage = "";
    try {
      const job = await submitTrainingJob(buildRequest());
      successMessage = `Job ${job.id} accodato.`;
      selectedJobId = job.id;
      startEvents(job.id);
      await refreshJobs();
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
    } finally {
      loading = false;
    }
  }

  function startEvents(jobId: string) {
    eventSource?.close();
    eventSource = new EventSource(trainingEventsUrl(jobId));
    eventSource.onmessage = () => void refreshJobs();
    eventSource.onerror = () => eventSource?.close();
  }

  async function cancel(jobId: string) {
    try {
      await cancelTrainingJob(jobId);
      await refreshJobs();
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : String(error);
    }
  }

  function openWandb(job: TrainingJobStatus) {
    if (job.wandb_url) window.open(job.wandb_url, "_blank", "noopener,noreferrer");
  }
</script>

<aside class="training-sidebar">
  <header>
    <h2>Training</h2>
    <button class="close" onclick={onClose} aria-label="Chiudi training">✖</button>
  </header>

  {#if errorMessage}<div class="message error">{errorMessage}</div>{/if}
  {#if successMessage}<div class="message success">{successMessage}</div>{/if}

  <section>
    <h3>Dataset</h3>
    <label>Classe Python
      <select value={selectedDataset} onchange={(event) => {
        const target = datasets.find((item) => item.target === (event.currentTarget as HTMLSelectElement).value);
        if (target) selectDataset(target);
      }}>
        {#each datasets as dataset (dataset.target)}
          <option value={dataset.target}>{dataset.name}</option>
        {/each}
      </select>
    </label>
    {#if selectedDatasetInfo}
      {#each selectedDatasetInfo.parameters as parameter (parameter.name)}
        <label>{parameter.name}
          <input value={datasetParams[parameter.name] ?? ""} oninput={(event) => setDatasetParameter(parameter, event)} />
        </label>
      {/each}
    {/if}
    <div class="grid">
      <label>Batch size<input type="number" bind:value={batchSize} /></label>
      <label>Worker<input type="number" bind:value={numWorkers} /></label>
      <label>Train split<input type="number" step="0.01" bind:value={trainSize} /></label>
      <label>Seed<input type="number" bind:value={seed} /></label>
    </div>
  </section>

  <section>
    <h3>Hydra</h3>
    <label>Optimizer target<input bind:value={optimizerTarget} /></label>
    <div class="grid">
      <label>Learning rate<input type="number" step="0.0001" bind:value={learningRate} /></label>
      <label>Epochs<input type="number" bind:value={maxEpochs} /></label>
      <label>Accelerator<input bind:value={accelerator} /></label>
      <label>Patience<input type="number" bind:value={patience} /></label>
      <label>Min delta<input type="number" step="0.001" bind:value={minDelta} /></label>
    </div>
    <label>Override Hydra (una per riga)
      <textarea bind:value={overridesText} placeholder="trainer.max_epochs=10"></textarea>
    </label>
  </section>

  <section>
    <h3>W&B</h3>
    <div class="grid">
      <label>Project<input bind:value={wandbProject} /></label>
      <label>Mode<input bind:value={wandbMode} /></label>
    </div>
  </section>

  <section>
    <h3>Risorse e priorità</h3>
    <div class="grid">
      <label>CPU<input type="number" bind:value={cpu} /></label>
      <label>RAM GB<input type="number" bind:value={memoryGb} /></label>
      <label>GPU<input type="number" bind:value={gpu} /></label>
      <label>GPU RAM GB<input type="number" bind:value={gpuMemoryGb} /></label>
    </div>
    <label>Tipo GPU<input bind:value={gpuType} placeholder="A100" /></label>
    <label>Nodo<input bind:value={node} placeholder="qualsiasi" /></label>
    <label>Priorità<input type="number" bind:value={priority} /></label>
    <button class="submit" onclick={submit} disabled={loading}>{loading ? "Invio..." : "Invia training"}</button>
  </section>

  <section class="jobs">
    <h3>Job {#if loadingJobs}…{/if}</h3>
    {#each jobs as job (job.id)}
      <article class:selected={selectedJobId === job.id}>
        <button class="job-title" onclick={() => { selectedJobId = job.id; startEvents(job.id); }}>
          <span>{job.id.slice(0, 8)}</span><strong>{job.status}</strong>
        </button>
        <small>priorità {job.priority} · {job.executor ?? "in coda"}</small>
        {#if job.error}<pre>{job.error}</pre>{/if}
        {#if job.status === "running"}<button onclick={() => cancel(job.id)}>Annulla</button>{/if}
        {#if job.wandb_url}<button onclick={() => openWandb(job)}>Apri W&B</button>{/if}
      </article>
    {:else}
      <p>Nessun job.</p>
    {/each}
  </section>
</aside>

<style>
  .training-sidebar { position: fixed; z-index: 20; top: 0; right: 0; bottom: 0; width: min(410px, 100vw); overflow-y: auto; padding: 18px; background: #fff; box-shadow: -4px 0 18px #0002; font-family: sans-serif; }
  header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #ddd; margin-bottom: 12px; }
  h2, h3 { margin: 0 0 10px; } h3 { font-size: 1rem; }
  section { border-bottom: 1px solid #e5e7eb; padding: 12px 0; }
  label { display: flex; flex-direction: column; gap: 4px; margin: 7px 0; font-size: .82rem; }
  input, select, textarea { box-sizing: border-box; width: 100%; padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; background: #fff; }
  textarea { min-height: 62px; font-family: monospace; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; } .grid label { margin: 0; }
  button { padding: 6px 9px; border: 1px solid #cbd5e1; border-radius: 4px; background: #f8fafc; cursor: pointer; }
  button:hover { background: #e2e8f0; } button:disabled { cursor: wait; opacity: .6; }
  .close { border: 0; background: transparent; font-size: 1.1rem; } .submit { width: 100%; margin-top: 8px; background: #2563eb; color: white; border-color: #2563eb; }
  .message { padding: 8px; border-radius: 4px; margin-bottom: 8px; font-size: .82rem; } .error { color: #991b1b; background: #fee2e2; } .success { color: #166534; background: #dcfce7; }
  article { margin: 7px 0; padding: 8px; border: 1px solid #e2e8f0; border-radius: 5px; } article.selected { border-color: #2563eb; }
  .job-title { width: 100%; display: flex; justify-content: space-between; } small { color: #64748b; } pre { max-height: 100px; overflow: auto; white-space: pre-wrap; color: #991b1b; font-size: .72rem; }
</style>
