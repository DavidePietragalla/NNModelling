"""
Convert Visual Paradigm .vpp (SQLite) to PlantUML class diagram.
Usage: python3 vpp2plantuml.py [--output diagram.puml]

Parses VP SQLite format to extract: classes, generalizations, associations, multiplicities.
Output is valid PlantUML.
"""

import sqlite3
import re
import sys

DB = "/home/softdream/Programming/gits/NNModelling/analysis/uml/nn.vpp"
NODE_ROOT_ID = "kzVFHzmGAqACOAnG"


def clean(name):
    return name.replace("\\", "").replace('"', "").strip()


def extract_assoc_ends(raw):
    """Extract (classID, multiplicity) for from/to ends of an association.

    VP format:
        from={ID:"role":AssociationEnd { ... EndModelElement=<c:classID> ... multiplicity="1" ... }}
    """
    ends = []
    for block_name in ("from", "to"):
        idx = raw.find(f"{block_name}={{")
        if idx == -1:
            continue
        # Find the matching closing }} for this block
        # AssociationEnd blocks end with }}} — the inner content uses single }.
        # We find the first }} after the opening { that ends the block.
        start = raw.index("{", idx) + 1
        depth = 1
        pos = start
        while depth > 0 and pos < len(raw):
            if raw[pos] == "{":
                depth += 1
            elif raw[pos] == "}":
                depth -= 1
            pos += 1
        block = raw[start:pos-1]

        em = re.search(r'EndModelElement=<([^>]+)>', block)
        mm = re.search(r'multiplicity="([^"]*)"', block)
        if em:
            eid = em.group(1).split(":")[-1]
            mult = mm.group(1) if mm else ""
            ends.append((eid, mult))
    return ends


def main():
    out_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        out_path = sys.argv[idx + 1]

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # --- Classes: deduplicate by name ---
    seen_names = set()
    class_id_to_name = {}
    cur.execute(
        "SELECT ID, NAME FROM MODEL_ELEMENT WHERE MODEL_TYPE='Class' AND NAME IS NOT NULL AND NAME NOT LIKE 'Class%' ORDER BY NAME"
    )
    for cid, raw_name in cur.fetchall():
        name = clean(raw_name)
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        class_id_to_name[cid] = name

    # --- Generalizations (who extends Node) ---
    node_children_ids = []
    cur.execute("SELECT DEFINITION FROM MODEL_ELEMENT WHERE MODEL_TYPE='Generalization'")
    for (blob,) in cur.fetchall():
        raw = blob.decode("latin-1")
        m = re.search(r"toModel=<([^:]+):([^>]+)>", raw)
        if m and m.group(2) in class_id_to_name:
            node_children_ids.append(m.group(2))

    # --- Associations ---
    assocs = []
    cur.execute(
        "SELECT e.NAME, e.DEFINITION FROM MODEL_ELEMENT e WHERE e.MODEL_TYPE='Association' AND e.NAME IS NOT NULL AND e.NAME != ''"
    )
    for name, blob in cur.fetchall():
        raw = blob.decode("latin-1")
        label = clean(name)
        ends = extract_assoc_ends(raw)
        if len(ends) == 2:
            c1, m1 = ends[0]
            c2, m2 = ends[1]
            if c1 in class_id_to_name and c2 in class_id_to_name and c1 != c2:
                assocs.append((c1, c2, label, m1, m2))

    # --- Generate PlantUML ---
    node_children = {cid for cid in node_children_ids if cid in class_id_to_name}
    non_node_ids = [
        cid for cid in class_id_to_name
        if cid not in node_children and cid != NODE_ROOT_ID
        and class_id_to_name[cid] != "Node"  # avoid duplicate with abstract Node
    ]

    lines = []
    lines.append("@startuml")
    lines.append("")
    lines.append("skinparam classAttributeIconSize 0")
    lines.append("skinparam backgroundColor #FEFEFE")
    lines.append("")
    lines.append("' === NNModelling Metamodel (generated from Visual Paradigm .vpp) ===")
    lines.append("")

    lines.append("abstract class Node {")
    lines.append("  +id: string")
    lines.append("  +name: string")
    lines.append("  +type: string")
    lines.append("}")
    lines.append("")

    for cid in sorted(node_children, key=lambda c: class_id_to_name[c]):
        name = class_id_to_name[cid]
        lines.append(f"class {name} {{")
        lines.append("}")
        lines.append(f"Node <|-- {name}")
        lines.append("")

    for cid in sorted(non_node_ids, key=lambda c: class_id_to_name[c]):
        name = class_id_to_name[cid]
        lines.append(f"class {name} {{")
        lines.append("}")
        lines.append("")

    for src_id, tgt_id, label, m_src, m_tgt in assocs:
        src = class_id_to_name[src_id]
        tgt = class_id_to_name[tgt_id]
        # Clean multiplicity — VP sometimes stores "1 {id}" for qualified assocs
        m_src_clean = m_src.split("{")[0].strip()
        m_tgt_clean = m_tgt.split("{")[0].strip()
        lines.append(f'{src} "{m_src_clean}" --> "{m_tgt_clean}" {tgt} : {label}')

    lines.append("")
    lines.append("hide empty members")
    lines.append("")
    lines.append("@enduml")

    output = "\n".join(lines)

    if out_path:
        with open(out_path, "w") as f:
            f.write(output)
        print(f"Written to {out_path}")
    else:
        print(output)

    conn.close()


if __name__ == "__main__":
    main()
