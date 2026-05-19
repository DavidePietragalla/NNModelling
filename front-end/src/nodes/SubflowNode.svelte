<script lang="ts">
  import { Handle, Position, NodeResizer, type NodeProps } from "@xyflow/svelte";
  import { type Node, type OnResizeEnd } from "@xyflow/svelte";

  type SubflowData = {
    label: string;
    isCollapsed: boolean;
    color: any,
    params: Object,
    stereotype: any,
    onToggle: (id: string, collapse: boolean) => void;
    onResizeEnd: (id: string, width: number, height: number) => void;
  };

  type MySubflowNode = Node<SubflowData, 'subflow'>;

  let { data, selected, id }: NodeProps<MySubflowNode> = $props();

  let topParams = $derived(
    Object.entries(data.params || {}).filter(
      ([_, p]: any) => p?.position === "top",
    ),
  );

  let bottomParams = $derived(
    Object.entries(data.params || {}).filter(
      ([_, p]: any) => p?.position === "bottom",
    ),
  );

  const handleResize: OnResizeEnd = (event, params) => {
    data.onResizeEnd(id, params.width, params.height);
  };
</script>

<NodeResizer minWidth={200} minHeight={50} isVisible={selected} onResizeEnd={handleResize}/>

<Handle type="target" position={Position.Top} />

<div class="subflow-wrapper" class:collapsed={data.isCollapsed}>
  <div class="subflow-header" style:background={String(data.color || "#007bff")}>
    {data.label?.slice(11) || ''}
    <button class="collapse-btn" onclick={() => data.onToggle(id, !data.isCollapsed)}>
      {data.isCollapsed ? '+' : '-'}
    </button>
  </div>

  {#if data.stereotype && topParams.length > 0}
    <div class="params-container top-params">
      {#each topParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}

  <div class="subflow-body"></div>

  {#if data.stereotype && bottomParams.length > 0}
    <div class="params-container bottom-params">
      {#each bottomParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<Handle type="source" position={Position.Bottom} />

<style>
  @import "../styles/subflow.css";
</style>
