"""
Extract UML model elements from Visual Paradigm .vpp file (SQLite format).
Usage: python3 vpp_extract.py [--detailed]

Dumps class hierarchy, associations, generalizations, and stereotypes.
Use --detailed to also show raw attribute definitions per class.
"""

import sqlite3
import re
import sys

DB = "/home/softdream/Programming/gits/NNModelling/analysis/uml/nn.vpp"


def main():
    detailed = "--detailed" in sys.argv
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # --- CLASSES ---
    print("=" * 60)
    print("CLASSES")
    print("=" * 60)
    cur.execute(
        "SELECT ID, NAME FROM MODEL_ELEMENT WHERE MODEL_TYPE='Class' AND NAME IS NOT NULL AND NAME NOT LIKE 'Class%' ORDER BY NAME"
    )
    classes = {row[0]: row[1] for row in cur.fetchall()}
    for cid, name in sorted(classes.items(), key=lambda x: x[1] or ""):
        print(f"  {name:<30s} ({cid})")
    print(f"\n  Total: {len(classes)} classes")

    # --- GENERALIZATIONS (inheritance) ---
    print("\n" + "=" * 60)
    print("GENERALIZATION HIERARCHY")
    print("=" * 60)
    cur.execute(
        "SELECT DEFINITION FROM MODEL_ELEMENT WHERE MODEL_TYPE='Generalization'"
    )
    gen_count = 0
    for (blob,) in cur.fetchall():
        raw = blob.decode("latin-1")
        m = re.search(r"toModel=<([^:]+):([^>]+)>", raw)
        if m:
            child_id = m.group(2)
            child_name = classes.get(child_id, f"? ({child_id})")
            print(f"  {child_name} -> Node")
            gen_count += 1
    print(f"\n  Total: {gen_count} generalizations")

    # --- ASSOCIATIONS ---
    print("\n" + "=" * 60)
    print("ASSOCIATIONS")
    print("=" * 60)
    cur.execute(
        "SELECT e.NAME, e.DEFINITION FROM MODEL_ELEMENT e WHERE e.MODEL_TYPE='Association' AND e.NAME IS NOT NULL"
    )
    for name, blob in cur.fetchall():
        raw = blob.decode("latin-1") if isinstance(blob, bytes) else blob
        # Extract end references (VP internal IDs)
        ends = re.findall(r"<[^:]+:[^:]+:([^>]+)>", raw)
        end_names = []
        for eid in ends:
            en = classes.get(eid, eid[:24])
            end_names.append(en)
        display = name if name else "(unnamed)"
        print(f"  {display:<30s} {end_names}")

    # --- STEREOTYPES ---
    print("\n" + "=" * 60)
    print("STEREOTYPES (UML Profile)")
    print("=" * 60)
    cur.execute(
        "SELECT e.NAME, e.DEFINITION FROM MODEL_ELEMENT e WHERE e.MODEL_TYPE='Stereotype'"
    )
    for name, blob in cur.fetchall():
        raw = blob.decode("latin-1")
        t = re.search(r'type_string="([^"]+)"', raw)
        stype = t.group(1) if t else "?"
        print(f"  {name:<30s} -> {stype}")

    # --- DETAILED: class attributes ---
    if detailed:
        print("\n" + "=" * 60)
        print("CLASS ATTRIBUTES (raw)")
        print("=" * 60)
        cur.execute(
            "SELECT NAME, DEFINITION FROM MODEL_ELEMENT WHERE MODEL_TYPE='Class' AND NAME IS NOT NULL AND NAME NOT LIKE 'Class%' ORDER BY NAME"
        )
        for name, blob in cur.fetchall():
            raw = blob.decode("latin-1")
            print(f"\n--- {name} ---")
            # Extract content between { and last }
            m = re.search(r"\{(.+)\}", raw, re.DOTALL)
            if m:
                for line in m.group(1).split("\n"):
                    line = line.strip()
                    if line:
                        print(f"  {line[:120]}")

    conn.close()


if __name__ == "__main__":
    main()
