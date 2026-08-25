# Scouting inbox

AI-scout output JSON files land here before import, e.g.

- `luna_gw01_main_2026-08-20T1800Z.json`
- `glm_gw01_main_2026-08-20T1800Z.json`

The importer is agent-neutral: the JSON's `agent` field records provenance.
Workflow per file:

```bash
python scripts/validate_scouting.py data/scouting/inbox/<file>.json
python scripts/import_scouting.py data/scouting/inbox/<file>.json --dry-run
python scripts/import_scouting.py data/scouting/inbox/<file>.json
```

Every pass must use a unique filename and genuinely new content: the importer
blocks an identical file hash unless `--force` is supplied.
