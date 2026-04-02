#!/usr/bin/env python3
"""Generate an HTML dashboard from E2E test results.

Reads tests/e2e/results/report.json and produces an interactive HTML dashboard
at tests/e2e/results/dashboard.html.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --input path/to/report.json --output path/to/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, "tests", "e2e", "results", "report.json")
DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "tests", "e2e", "results", "dashboard.html")

# Quality thresholds from README
THRESHOLDS = {
    "fraud_f1": {"min": 0.70, "label": "Fraud F1 Score"},
    "escalation_rate": {"max": 0.10, "label": "Escalation Rate"},
    "false_positive_rate": {"max": 0.05, "label": "False Positive Rate"},
    "aml_structuring_recall": {"min": 0.60, "label": "AML Structuring Recall"},
}


def _status_badge(key: str, value: float) -> tuple[str, str, str]:
    """Return (status_text, color, icon) for a metric against its threshold."""
    threshold = THRESHOLDS.get(key)
    if not threshold:
        return "N/A", "#888", "&#8226;"

    if "min" in threshold:
        passed = value >= threshold["min"]
        target = f"&ge; {threshold['min']:.0%}"
    else:
        passed = value <= threshold["max"]
        target = f"&le; {threshold['max']:.0%}"

    if passed:
        return f"PASS ({target})", "#22c55e", "&#10003;"
    else:
        return f"FAIL ({target})", "#ef4444", "&#10007;"


def generate_html(results: dict) -> str:
    """Build a self-contained HTML dashboard string from E2E results."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total = results["total_transactions"]
    fraud_total = results["total_fraud"]
    aml_total = results["total_aml_users"]

    # Build threshold cards
    threshold_cards = ""
    for key, meta in THRESHOLDS.items():
        value = results.get(key, 0.0)
        status_text, color, icon = _status_badge(key, value)
        threshold_cards += f"""
        <div class="card threshold-card" style="border-left: 4px solid {color};">
            <div class="card-icon" style="color: {color};">{icon}</div>
            <div class="card-body">
                <div class="card-title">{meta['label']}</div>
                <div class="card-value">{value:.1%}</div>
                <div class="card-status" style="color: {color};">{status_text}</div>
            </div>
        </div>"""

    # AML typology rows
    aml_typologies = [
        ("Structuring", results.get("aml_structuring_recall", 0)),
        ("Smurfing", results.get("aml_smurfing_recall", 0)),
        ("Layering", results.get("aml_layering_recall", 0)),
        ("Round-Tripping", results.get("aml_round_tripping_recall", 0)),
        ("Profile Mismatch", results.get("aml_profile_mismatch_recall", 0)),
    ]

    aml_rows = ""
    for name, recall in aml_typologies:
        bar_width = recall * 100
        bar_color = "#22c55e" if recall >= 0.60 else "#f59e0b" if recall >= 0.40 else "#ef4444"
        aml_rows += f"""
            <tr>
                <td>{name}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width: {bar_width}%; background: {bar_color};"></div>
                    </div>
                </td>
                <td class="num">{recall:.1%}</td>
            </tr>"""

    # Routing breakdown for donut chart (CSS-only)
    approve_pct = results.get("auto_approve_rate", 0) * 100
    escalate_pct = results.get("escalation_rate", 0) * 100
    block_pct = results.get("block_rate", 0) * 100

    # Confusion matrix
    tp = results.get("tp", 0)
    fp = results.get("fp", 0)
    fn = results.get("fn", 0)
    tn = results.get("tn", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection E2E Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0f172a;
        color: #e2e8f0;
        padding: 24px;
        line-height: 1.5;
    }}
    .header {{
        text-align: center;
        margin-bottom: 32px;
        padding-bottom: 16px;
        border-bottom: 1px solid #334155;
    }}
    .header h1 {{ font-size: 28px; color: #f8fafc; margin-bottom: 4px; }}
    .header .subtitle {{ color: #94a3b8; font-size: 14px; }}
    .header .timestamp {{ color: #64748b; font-size: 12px; margin-top: 8px; }}
    .section {{ margin-bottom: 32px; }}
    .section h2 {{
        font-size: 18px; color: #f8fafc; margin-bottom: 16px;
        padding-bottom: 8px; border-bottom: 1px solid #1e293b;
    }}
    .grid {{ display: grid; gap: 16px; }}
    .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
    .card {{
        background: #1e293b;
        border-radius: 8px;
        padding: 20px;
    }}
    .threshold-card {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .card-icon {{ font-size: 28px; font-weight: bold; }}
    .card-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .card-value {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
    .card-status {{ font-size: 12px; font-weight: 600; }}

    .stat-card .card-value {{ font-size: 32px; }}
    .stat-card .card-label {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}

    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; font-size: 12px; color: #94a3b8; text-transform: uppercase;
          letter-spacing: 0.5px; padding: 8px 12px; border-bottom: 1px solid #334155; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 14px; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }}

    .bar-container {{
        background: #334155;
        border-radius: 4px;
        height: 20px;
        width: 100%;
        overflow: hidden;
    }}
    .bar {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }}

    .confusion-grid {{
        display: grid;
        grid-template-columns: 100px 1fr 1fr;
        grid-template-rows: auto auto auto;
        gap: 4px;
        max-width: 360px;
    }}
    .confusion-header {{
        font-size: 11px; color: #94a3b8; text-align: center;
        padding: 8px; text-transform: uppercase;
    }}
    .confusion-cell {{
        padding: 16px;
        border-radius: 6px;
        text-align: center;
        font-weight: 700;
        font-size: 20px;
    }}
    .confusion-label {{
        font-size: 10px; font-weight: 400; color: #94a3b8;
        display: block; margin-top: 4px;
    }}
    .tp {{ background: #166534; color: #bbf7d0; }}
    .fp {{ background: #991b1b; color: #fecaca; }}
    .fn {{ background: #92400e; color: #fde68a; }}
    .tn {{ background: #1e3a5f; color: #bfdbfe; }}

    .routing-bar {{
        display: flex;
        height: 36px;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 12px;
    }}
    .routing-bar div {{ display: flex; align-items: center; justify-content: center;
                        font-size: 12px; font-weight: 600; color: #fff; }}
    .routing-legend {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .routing-legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
    .routing-legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}

    .cost-row {{ display: flex; justify-content: space-between; padding: 8px 0;
                 border-bottom: 1px solid #1e293b; }}
    .cost-row:last-child {{ border-bottom: none; }}
    .cost-label {{ color: #94a3b8; }}
    .cost-value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

    .overall-pass {{ text-align: center; padding: 16px; border-radius: 8px;
                     font-weight: 700; font-size: 16px; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Fraud Detection Agent &mdash; E2E Results</h1>
        <div class="subtitle">
            {total:,} transactions &bull; {fraud_total:,} fraud labels &bull; {aml_total} AML-injected users
        </div>
        <div class="timestamp">Generated: {now}</div>
    </div>

    <!-- Quality Thresholds -->
    <div class="section">
        <h2>Quality Thresholds</h2>
        <div class="grid grid-4">{threshold_cards}
        </div>
    </div>

    <!-- Fraud Detection -->
    <div class="section">
        <h2>Fraud Detection Performance</h2>
        <div class="grid grid-3">
            <div class="card stat-card">
                <div class="card-title">Precision</div>
                <div class="card-value">{results.get('fraud_precision', 0):.1%}</div>
                <div class="card-label">Of blocked transactions, how many were truly fraud</div>
            </div>
            <div class="card stat-card">
                <div class="card-title">Recall</div>
                <div class="card-value">{results.get('fraud_recall', 0):.1%}</div>
                <div class="card-label">Of actual fraud, how many were caught</div>
            </div>
            <div class="card stat-card">
                <div class="card-title">F1 Score</div>
                <div class="card-value">{results.get('fraud_f1', 0):.1%}</div>
                <div class="card-label">Harmonic mean of precision and recall</div>
            </div>
        </div>
    </div>

    <!-- Confusion Matrix + Routing -->
    <div class="section">
        <div class="grid grid-2">
            <div class="card">
                <div class="card-title" style="margin-bottom: 16px;">Confusion Matrix</div>
                <div class="confusion-grid">
                    <div></div>
                    <div class="confusion-header">Predicted Fraud</div>
                    <div class="confusion-header">Predicted Legit</div>
                    <div class="confusion-header" style="text-align: right;">Actual Fraud</div>
                    <div class="confusion-cell tp">{tp}<span class="confusion-label">True Pos</span></div>
                    <div class="confusion-cell fn">{fn}<span class="confusion-label">False Neg</span></div>
                    <div class="confusion-header" style="text-align: right;">Actual Legit</div>
                    <div class="confusion-cell fp">{fp}<span class="confusion-label">False Pos</span></div>
                    <div class="confusion-cell tn">{tn}<span class="confusion-label">True Neg</span></div>
                </div>
            </div>
            <div class="card">
                <div class="card-title" style="margin-bottom: 16px;">Routing Breakdown</div>
                <div class="routing-bar">
                    <div style="width: {approve_pct}%; background: #22c55e;">{approve_pct:.1f}%</div>
                    <div style="width: {escalate_pct}%; background: #f59e0b;">{escalate_pct:.1f}%</div>
                    <div style="width: {block_pct}%; background: #ef4444;">{block_pct:.1f}%</div>
                </div>
                <div class="routing-legend">
                    <div class="routing-legend-item">
                        <div class="routing-legend-dot" style="background: #22c55e;"></div>
                        Approved ({results.get('approve_count', 0):,})
                    </div>
                    <div class="routing-legend-item">
                        <div class="routing-legend-dot" style="background: #f59e0b;"></div>
                        Escalated ({results.get('escalation_count', 0):,})
                    </div>
                    <div class="routing-legend-item">
                        <div class="routing-legend-dot" style="background: #ef4444;"></div>
                        Blocked ({results.get('block_count', 0):,})
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- AML Detection -->
    <div class="section">
        <h2>AML Detection by Typology</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th style="width: 180px;">Typology</th>
                        <th>Recall</th>
                        <th style="width: 80px; text-align: right;">Rate</th>
                    </tr>
                </thead>
                <tbody>{aml_rows}
                </tbody>
            </table>
            <div style="margin-top: 16px; font-size: 13px; color: #94a3b8;">
                Overall AML recall: <strong style="color: #f8fafc;">{results.get('aml_overall_recall', 0):.1%}</strong>
                &bull; Investigations opened: <strong style="color: #f8fafc;">{results.get('aml_investigations_opened', 0)}</strong>
                &bull; Correct matches: <strong style="color: #f8fafc;">{results.get('aml_investigations_correct', 0)}/{aml_total}</strong>
            </div>
        </div>
    </div>

    <!-- Cost -->
    <div class="section">
        <h2>Cost &amp; Token Usage</h2>
        <div class="card" style="max-width: 400px;">
            <div class="cost-row">
                <span class="cost-label">Total tokens</span>
                <span class="cost-value">{results.get('total_tokens', 0):,}</span>
            </div>
            <div class="cost-row">
                <span class="cost-label">Estimated cost</span>
                <span class="cost-value">${results.get('estimated_cost', 0):.2f}</span>
            </div>
            <div class="cost-row">
                <span class="cost-label">Avg tokens / transaction</span>
                <span class="cost-value">{results.get('total_tokens', 0) // max(total, 1):,}</span>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E2E results dashboard")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to report.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: report file not found at {args.input}", file=sys.stderr)
        print("Run the E2E tests first: pytest tests/e2e/test_runner.py -v -s", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    html = generate_html(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard written to: {args.output}")


if __name__ == "__main__":
    main()
