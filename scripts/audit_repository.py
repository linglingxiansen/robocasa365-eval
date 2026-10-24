"""Scan a source snapshot for secrets, private infrastructure and unsafe files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


RULES = {
    "private-storage": re.compile(r"/(?:mnt/[^\s\"']*(?:dolphinfs|docker/user)|home/hadoop[^\s\"']*)", re.I),
    "internal-domain": re.compile(r"\b(?:[\w.-]+\.)?(?:sankuai|meituan)\.(?:com|net)\b", re.I),
    "internal-user": re.compile(r"\bhadoop-[a-z0-9_.-]+\b", re.I),
    "private-ipv4": re.compile(r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "hub-token": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    "aws-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}
UNSAFE_SUFFIXES = {".pem", ".key", ".pt", ".pth", ".safetensors", ".pyc", ".log"}


def audit(root, forbidden=(), *, include_artifacts=False):
    root = Path(root).resolve()
    findings = []
    inspected = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if not include_artifacts and relative.parts[0] in {"artifacts", "outputs"}:
            continue
        if path.is_symlink():
            findings.append({"path": str(relative), "rule": "symlink"})
            continue
        if not path.is_file():
            continue
        if path.suffix in UNSAFE_SUFFIXES or path.name.startswith(".env"):
            findings.append({"path": str(relative), "rule": "unsafe-file"})
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            findings.append({"path": str(relative), "rule": "unreviewed-binary"})
            continue
        inspected += 1
        for name, rule in RULES.items():
            for match in rule.finditer(text):
                findings.append({"path": str(relative), "rule": name,
                                 "line": text.count("\n", 0, match.start())+1})
        for i, value in enumerate(forbidden):
            if value.lower() in text.lower() or value.lower() in str(relative).lower():
                findings.append({"path": str(relative), "rule": f"custom-identifier-{i+1}"})
    return {"files_scanned": inspected, "passed": not findings, "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--forbid", action="append", default=[], help="Additional local-only identifier to reject")
    parser.add_argument("--include-artifacts", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, args.forbid, include_artifacts=args.include_artifacts)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
