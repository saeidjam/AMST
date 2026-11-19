"""Adapter template connecting DisCoDisCo to AMST's EJD backend.

AMST calls this script (configured as metrics.ejd.discodisco_cmd) with JSONL on
stdin, one object per response:   {"id": 0, "text": "..."}
and expects JSONL on stdout:        {"id": 0, "n_units": 7, "relations": ["causal-cause", "elaboration", ...]}

`relations` lists the relation label of each link between consecutive EDUs
(N_U - 1 labels). AMST counts a link as justificatory when its label contains
cause, result, condition, or concession.

Setup (once):
  git clone https://github.com/gucorpling/DisCoDisCo
  follow its README to download a trained segmenter and relation classifier
  (an English RST corpus model, e.g. eng.rst.gum), then implement the two
  functions below with DisCoDisCo's prediction scripts.

Config:
  metrics:
    ejd:
      backend: discodisco
      discodisco_cmd: "python scripts/discodisco_adapter.py"
"""
import json
import sys


def segment(text: str) -> list[str]:
    """Return the EDUs of `text` using the DisCoDisCo segmenter."""
    raise NotImplementedError("call the DisCoDisCo segmenter here")


def classify(units: list[str]) -> list[str]:
    """Return one relation label for each pair of consecutive EDUs."""
    raise NotImplementedError("call the DisCoDisCo relation classifier here")


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        item = json.loads(line)
        units = segment(item["text"])
        rels = classify(units) if len(units) > 1 else []
        print(json.dumps({"id": item["id"], "n_units": len(units), "relations": rels}))


if __name__ == "__main__":
    main()
