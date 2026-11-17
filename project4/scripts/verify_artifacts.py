"""Check public scientific artifacts and report identities without private data."""

import hashlib
import importlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    for module, expected in manifest["source_sha256"].items():
        if digest(Path(importlib.import_module(module).__file__)) != expected:
            raise ValueError(f"Source differs from the recorded calculation: {module}")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(results / name) != expected:
            raise ValueError(f"Numerical artifact differs from its recorded identity: {name}")
    notebook = json.loads((results / "notebook_validation.json").read_text())
    if digest(root / notebook["notebook"]) != notebook["sha256"]:
        raise ValueError("Notebook differs from its execution record")
    if digest(results / "manifest.json") != notebook["numerical_manifest_sha256"]:
        raise ValueError("Re-execute the notebook after changing the numerical study")
    report = json.loads((results / "report_validation.json").read_text())
    if digest(root / report["output"]) != report["pdf_sha256"]:
        raise ValueError("PDF differs from its recorded identity")
    for name, expected in report["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Rebuild the PDF after changing its source: {name}")
    for name, expected in report["builder_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Rebuild the PDF after changing its builder: {name}")
    print(
        f"Verified {len(manifest['artifact_sha256'])} numerical artifacts, shared package sources, executed notebook and PDF."
    )


if __name__ == "__main__":
    main()
