# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""Download exam datasets into data/raw/."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    csv_url = (
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/"
        "Ghana_Election_Result.csv"
    )
    pdf_url = (
        "https://mofep.gov.gh/sites/default/files/"
        "budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    )
    csv_path = RAW / "Ghana_Election_Result.csv"
    pdf_path = RAW / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

    print("Fetching CSV…")
    r = requests.get(csv_url, timeout=120)
    r.raise_for_status()
    csv_path.write_bytes(r.content)
    print("Wrote", csv_path)

    print("Fetching PDF (may be large)…")
    r = requests.get(pdf_url, timeout=300)
    r.raise_for_status()
    pdf_path.write_bytes(r.content)
    print("Wrote", pdf_path)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
