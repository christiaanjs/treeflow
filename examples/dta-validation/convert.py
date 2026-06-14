"""Convert LPhy nexus outputs to treeflow_dta_hmc inputs.

Reads:
  dta-validation_D.nexus      -- single-site 'standard' alignment
  dta-validation_psi.trees    -- nexus trees block (one tree)

Writes:
  traits.csv                  -- taxon,trait (header), one row per tip
  tree.nwk                    -- a single newick line
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_nexus_alignment(path: Path) -> dict[str, str]:
    text = path.read_text()
    m = re.search(r"matrix\s*(.*?)\s*;", text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError(f"No matrix block in {path}")
    mapping: dict[str, str] = {}
    for line in m.group(1).strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        taxon, state = parts[0], parts[1]
        mapping[taxon] = state
    return mapping


def parse_nexus_tree(path: Path) -> str:
    text = path.read_text()
    m = re.search(
        r"^\s*tree\s+\S+\s*=\s*(?:\[[^\]]*\]\s*)?(.+?;)\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"No tree statement found in {path}")
    return m.group(1).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--prefix", default="dta-validation")
    args = ap.parse_args()

    traits = parse_nexus_alignment(args.indir / f"{args.prefix}_D.nexus")
    newick = parse_nexus_tree(args.indir / f"{args.prefix}_psi.trees")

    traits_path = args.indir / "traits.csv"
    with traits_path.open("w") as fh:
        fh.write("taxon,trait\n")
        for taxon in sorted(traits, key=lambda s: int(s) if s.isdigit() else s):
            fh.write(f"{taxon},{traits[taxon]}\n")

    tree_path = args.indir / "tree.nwk"
    tree_path.write_text(newick + "\n")

    counts: dict[str, int] = {}
    for v in traits.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"Wrote {traits_path} ({len(traits)} taxa)")
    print(f"Wrote {tree_path}")
    print(f"State counts: {sorted(counts.items())}")


if __name__ == "__main__":
    main()
