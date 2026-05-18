<script lang="ts">
  import { NodeResizer, type NodeProps, type Node } from "@xyflow/svelte";

  type SubflowData = {
    label: string;
    isCollapsed: boolean;
    onToggle: (id: string, collapse: boolean) => void;
  };

  type MySubflowNode = Node<SubflowData, 'subflow'>;

  let { data, selected, id }: NodeProps<MySubflowNode> = $props();
</script>

<NodeResizer minWidth={150} minHeight={50} isVisible={selected} />

<div class="subflow-wrapper">
  <div class="subflow-label">
    {data.label}
    <button class="collapse-btn" onclick={() => data.onToggle(id, !data.isCollapsed)}>
      {data.isCollapsed ? '+' : '-'}
    </button>
  </div>
</div>

<style>
  @import "../styles/subflow.css";
</style>
