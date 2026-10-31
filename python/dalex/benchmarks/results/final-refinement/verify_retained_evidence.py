"""Verify retained bytes, archived trials and local results without model runs.

Requires only Python's standard library; does not download expired CI artifacts
or repeat the tests/benchmarks whose captured results it checks.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    files = read(root / "evidence-manifest.json")["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and p.name != "evidence-manifest.json"}
    assert actual == set(files), (actual - set(files), set(files) - actual)
    for name, expected in files.items():
        content = (root / name).read_bytes()
        assert len(content) == expected["bytes"], name
        assert sha256(content) == expected["sha256"], name

    ci = read(root / "ci-validation.json")
    identity = ci["source_identity"]
    assert ci["jobs"] == 23
    assert all(j["conclusion"] == "success" for j in ci["job_results"])
    assert ci["junit_kinds"]["full"] == 15
    assert ci["gil_enabled_after_imports"] is False
    assert ci["runtime_import_checks"] == 21
    local = read(root / "local-validation.json")
    assert local["source_identity"] == identity
    for check in local["checks"].values():
        path = root / check["junit_filename"]
        assert sha256(path.read_bytes()) == check["junit_sha256"]
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        for key in ("tests", "errors", "failures", "skipped"):
            assert sum(int(s.get(key, 0)) for s in suites) == check[key]
        assert check["errors"] == check["failures"] == 0

    manifest = read(root / "benchmark-manifest.json")
    archive_path = root / manifest["archive"]["filename"]
    assert sha256(archive_path.read_bytes()) == manifest["archive"]["sha256"]
    workers = 0
    maximum = 0.0
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {r["archive_entry"] for r in manifest["reports"]}
        for entry in manifest["reports"]:
            content = archive.read(entry["archive_entry"])
            assert len(content) == entry["uncompressed_bytes"]
            assert sha256(content) == entry["sha256"]
            report = json.loads(content)
            assert report["passed"]
            assert report["settings"]["trials"] == 5
            candidate = {k: v for k, v in report["sources"]["candidate"].items()
                         if k != "revision"}
            assert candidate == identity
            assert len(report["trials"]) == entry["workers"]
            workers += len(report["trials"])
            maximum = max(maximum, *report["max_absolute_parity_error"].values())
            for trial in report["trials"]:
                assert trial["data_sha256"] == report["data_sha256"]
                assert trial["fitted_model_sha256"] == report["fitted_model_sha256"]
                source = {k: v for k, v in report["sources"][trial["source_name"]].items()
                          if k != "revision"}
                assert trial["source"] == source
    assert workers == manifest["measured_workers"] == 570
    assert maximum <= 1e-12
    print(json.dumps({"passed": True, "retained_files": len(files),
                      "benchmark_workers": workers,
                      "benchmark_max_absolute_error": maximum,
                      "source_identity": identity}, indent=2))


if __name__ == "__main__":
    main()
