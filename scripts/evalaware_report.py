from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_mask.evalaware import RESULTS_ROOT, aggregate_report, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate EvalAwareBench judge results.")
    parser.add_argument("judgments", nargs="+", help="judge JSONL files")
    parser.add_argument("--out", default=str(RESULTS_ROOT / "reports" / "summary.json"))
    args = parser.parse_args()

    records = []
    for path in args.judgments:
        records.extend(
            record for record in read_jsonl(Path(path)) if record.get("judge")
        )
    report = aggregate_report(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
