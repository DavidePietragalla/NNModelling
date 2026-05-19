<script lang="ts">
  import { NodeResizer, type NodeProps, type Node, type OnResizeEnd } from "@xyflow/svelte";

  type SubflowData = {
    label: string;
    isCollapsed: boolean;
    onToggle: (id: string, collapse: boolean) => void;
    onResizeEnd: (id: string, width: number, height: number) => void;
  };

  type MySubflowNode = Node<SubflowData, 'subflow'>;

  let { data, selected, id }: NodeProps<MySubflowNode> = $props();

  const handleResize: OnResizeEnd = (event, params) => {
    data.onResizeEnd(id, params.width, params.height);
  };
</script>

<NodeResizer minWidth={200} minHeight={50} isVisible={selected} onResizeEnd={handleResize}/>

<div class="subflow-wrapper" class:collapsed={data.isCollapsed}>
  <div class="subflow-label">
    {data.label?.slice(11) || ''}
    <button class="collapse-btn" onclick={() => data.onToggle(id, !data.isCollapsed)}>
      {data.isCollapsed ? '+' : '-'}
    </button>
  </div>
</div>

<style>
  @import "../styles/subflow.css";
</style>
