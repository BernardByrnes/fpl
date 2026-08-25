# Decisions

Use the small CLI to keep human strategy separate from official facts:

```bash
python scripts/notes.py decision add --event 1 --action CAPTAIN --captain-id 123 --confidence medium --reasoning "..."
python scripts/notes.py decision list
python scripts/notes.py decision review 1 --notes "Review process and assumptions here."
```

Assumptions and invalidators are JSON arrays. Decisions are append-only until a review is added.

