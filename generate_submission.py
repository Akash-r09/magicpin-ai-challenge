"""
Generate canonical submission.jsonl for Magicpin Vera AI Challenge
==================================================================
Evaluates all 30 canonical (merchant, trigger) test pairs from test_pairs.json
and generates submission.jsonl adhering strictly to challenge-brief.md §7.2.
"""

from __future__ import annotations

import json
from pathlib import Path
import bot


def main():
    root = Path(__file__).parent
    dataset_dir = root / "dataset"
    pairs_file = dataset_dir / "expanded" / "test_pairs.json"
    
    if not pairs_file.exists():
        print(f"Error: {pairs_file} not found. Running generate_dataset.py first...")
        import subprocess
        subprocess.run(["python3", str(dataset_dir / "generate_dataset.py"), "--seed-dir", str(dataset_dir), "--out", str(dataset_dir / "expanded")], check=True)

    with open(pairs_file) as f:
        pairs = json.load(f)["pairs"]

    # Load categories
    categories = {}
    for cf in (dataset_dir / "categories").glob("*.json"):
        with open(cf) as f:
            c = json.load(f)
            categories[c["slug"]] = c

    # Load merchants from expanded or seeds
    merchants = {}
    m_dir = dataset_dir / "expanded" / "merchants"
    if m_dir.exists():
        for mf in m_dir.glob("*.json"):
            with open(mf) as f:
                m = json.load(f)
                merchants[m["merchant_id"]] = m
    else:
        with open(dataset_dir / "merchants_seed.json") as f:
            for m in json.load(f)["merchants"]:
                merchants[m["merchant_id"]] = m

    # Load customers from expanded or seeds
    customers = {}
    c_dir = dataset_dir / "expanded" / "customers"
    if c_dir.exists():
        for cf in c_dir.glob("*.json"):
            with open(cf) as f:
                c = json.load(f)
                customers[c["customer_id"]] = c
    else:
        with open(dataset_dir / "customers_seed.json") as f:
            for c in json.load(f)["customers"]:
                customers[c["customer_id"]] = c

    # Load triggers from expanded or seeds
    triggers = {}
    t_dir = dataset_dir / "expanded" / "triggers"
    if t_dir.exists():
        for tf in t_dir.glob("*.json"):
            with open(tf) as f:
                t = json.load(f)
                triggers[t["id"]] = t
    else:
        with open(dataset_dir / "triggers_seed.json") as f:
            for t in json.load(f)["triggers"]:
                triggers[t["id"]] = t

    submission_path = root / "submission.jsonl"
    lines = []

    for pair in pairs:
        t_id = pair["test_id"]
        trg_id = pair["trigger_id"]
        mid = pair["merchant_id"]
        cid = pair.get("customer_id")

        trg = triggers.get(trg_id)
        merch = merchants.get(mid)
        cust = customers.get(cid) if cid else None
        cat_slug = merch.get("category_slug", "") if merch else ""
        cat = categories.get(cat_slug, {})

        if not (trg and merch and cat):
            print(f"Warning: Missing context for {t_id} (merchant: {mid}, trigger: {trg_id})")
            continue

        result = bot.compose(cat, merch, trg, cust)
        submission_entry = {
            "test_id": t_id,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        }
        lines.append(json.dumps(submission_entry, ensure_ascii=False))

    with open(submission_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Successfully generated {len(lines)} submission lines to {submission_path}")


if __name__ == "__main__":
    main()
