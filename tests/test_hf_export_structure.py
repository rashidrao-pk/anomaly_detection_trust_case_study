import json
from pathlib import Path


def test_hf_export_documented_entrypoint():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "export_hf_torch.py").read_text()
    assert "torch.save(" in script
    assert '"config.json"' in script
    assert '"README.md"' in script
    assert "model.eval()" in script
