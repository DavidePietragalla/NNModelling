<!--
NNModelling — DSL for designing neural networks via visual node editor
Copyright (C) 2026  Luca Sforza

Licensed under the GNU General Public License v3 or later.
Commercial licenses are available — contact Luca Sforza.
See the LICENSE file for details.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-->

<script lang="ts">
  import type { Diagram } from "../Diagram.svelte";
  import type { StereotypeCore } from "../core/StereotypeCore";

  interface Props {
    selectedStereotype: StereotypeCore | null;
    diagram: Diagram;
    onSelectedChange: (stereotype: StereotypeCore | null) => void;
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
