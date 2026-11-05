"""Read-only hygiene gate over Git-tracked files (not a security certification)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    forbidden_suffixes = {".safetensors", ".gguf", ".sqlite3", ".db", ".pdf", ".npz", ".npy", ".csv", ".parquet", ".arrow", ".zip"}
    forbidden_parts = {"node_modules", ".venv", "__pycache__", "data", "artifacts", "dist", "test-results", "playwright-report"}
    patterns = {
        "personal absolute path": re.compile(r"/Users/[A-Za-z0-9_.-]+/|/opt/ana" + r"conda3/"),
        "private key": re.compile(r"-----" + r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "possible HF token": re.compile(r"hf_" + r"[A-Za-z0-9]{30,}"),
        "possible cloud key": re.compile(r"AKIA" + r"[A-Z0-9]{16}"),
        "possible API key": re.compile(r"sk-" + r"[A-Za-z0-9_-]{32,}"),
    }
    findings, total, largest = [], 0, 0
    for name in filter(None, tracked):
        relative = Path(name)
        path = root / relative
        if relative.suffix in forbidden_suffixes or set(relative.parts) & forbidden_parts:
            findings.append({"file":name,"category":"excluded artifact type/directory"})
        if path.is_symlink():
            findings.append({"file":name,"category":"tracked symlink needs manual review"})
            continue
        content = path.read_bytes()
        total += len(content)
        largest = max(largest, len(content))
        if len(content) > 5_000_000:
            findings.append({"file":name,"category":"file over 5 MB"})
        try:
            source = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for category, pattern in patterns.items():
            match = pattern.search(source)
            if match:
                # Never print the suspected credential itself.
                findings.append({"file":name,"line":source.count("\n",0,match.start())+1,"category":category})
    print(json.dumps({"tracked_files":len(list(filter(None,tracked))),"bytes":total,"largest_file_bytes":largest,
                      "findings":findings,"scope":"tracked current files; heuristic, not proof of absence"},indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
