<script lang="ts">
  import type { Diagram } from "../Diagram.svelte";
  import type { Stereotype } from "../stereotype";

  interface Props {
    selectedStereotype: Stereotype | null;
    diagram: Diagram;
    onSelectedChange: (stereotype: Stereotype | null) => void;
  }

  let { selectedStereotype, diagram, onSelectedChange }: Props = $props();
  // Funzione che intercetta il cambio nativo della select
  function handleChange(event: Event) {
    const select = event.target as HTMLSelectElement;
    const selectedName = select.value;

    if (!selectedName) {
      onSelectedChange(null);
      return;
    }

    // Cerchiamo l'oggetto Stereotype corretto e lo passiamo al padre
    const found = diagram.stereotypes.find((s) => s.name === selectedName);
    onSelectedChange(found || null);
  }
</script>

<select
  name="stereotypes"
  id="stereotypes"
  value={selectedStereotype ? selectedStereotype.name : ""}
  onchange={handleChange}
>
  <option value="">-- aggiungi layer --</option>

  {#each diagram.stereotypes as stereotype}
    <option value={stereotype.name}>
      {stereotype.name}
    </option>
  {/each}
</select>

<style>
  @import "../styles/dropdown.css";
</style>
