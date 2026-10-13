"""Prepare frozen training-case numerical RAG without EEG encoding or API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

RESEARCH_ROOT = Path(r"D:\桌面\文献总\EEGAGENT")
sys.path.insert(0, str(RESEARCH_ROOT))

from vrms_cloud.raw_case_rag import CASE_RAG_ROOT, prepare_case_banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CASE_RAG_ROOT)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    audit = prepare_case_banks(args.output)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
