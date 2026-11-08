import json

with open("LinguaBridge_AI_Development.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")

for i, cell in enumerate(nb.get('cells', [])):
    source = "".join(cell.get('source', []))
    keywords = ['gender', 'bias', 'benchmark', 'bleu', 'chrf', 'indic-parler', 'vyom', 'parler']
    matched = [k for k in keywords if k in source.lower()]
    if matched:
        print(f"Cell {i} ({cell.get('cell_type')}): Matched {matched}")
        lines = [l for l in source.split('\n') if any(k in l.lower() for k in keywords)]
        for l in lines[:3]:
            print("   ", l[:120])
