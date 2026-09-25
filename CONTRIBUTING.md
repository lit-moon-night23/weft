# Contributing to Weft

Weft is in its design phase. The scope and priorities are set out in README.md, and section 15 there lists where contributions help most.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Rules for every pull request

- **Tests pass.** Unit tests, engine-consistency checks, and the provenance lint must all pass.
- **Energy and area data carry provenance.** Each component entry cites its source paper and figure or table, the original technology node, and the scaling rule applied.
- **Presets state what they approximate.** A preset names the published design it is based on and lists what was simplified.
- **New mappers and noise models are plug-ins.** They go behind the documented interfaces and come with tests.
