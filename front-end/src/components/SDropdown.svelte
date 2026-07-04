<!--
NNModelling — DSL for designing neural networks via visual node editor
Copyright (C) 2026  Luca Sforza

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

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
