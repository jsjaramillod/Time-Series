"""Verify the recorded source, numerical, notebook, and PDF identities."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if digest(root / "src/shock_dynamics" / name) != expected:
            raise ValueError(f"Package source differs from recorded run: {name}")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(results / name) != expected:
            raise ValueError(f"Numerical artifact differs from recorded run: {name}")
    notebook = json.loads((results / "notebook_validation.json").read_text())
    if digest(root / notebook["notebook"]) != notebook["sha256"]:
        raise ValueError("Executed notebook differs from its validation record")
    if digest(results / "manifest.json") != notebook["numerical_manifest_sha256"]:
        raise ValueError("Re-execute the notebook after changing the numerical run")
    report = json.loads((results / "report_validation.json").read_text())
    if digest(root / report["output"]) != report["pdf_sha256"]:
        raise ValueError("PDF differs from its recorded identity")
    for name, expected in report["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Rebuild the PDF after changing its input: {name}")
    if digest(root / "scripts/build_research_report.py") != report["report_builder_sha256"]:
        raise ValueError("Rebuild the PDF after changing its builder")
    print(
        f"Verified {len(manifest['artifact_sha256'])} numerical artifacts, package sources, executed notebook, and PDF."
    )


if __name__ == "__main__":
    main()
