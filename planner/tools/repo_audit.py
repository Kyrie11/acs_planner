from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class Finding:
    severity: str
    title: str
    evidence: str
    recommendation: str


def _all_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts]


def _grep_text(files: Sequence[Path], pattern: str) -> list[Path]:
    hits = []
    regex = re.compile(pattern, re.MULTILINE)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if regex.search(text):
            hits.append(path)
    return hits


def _find_compute_trajectory_returns_nominal(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "compute_planner_trajectory" in text and ("nominal_traj" in text or "coarse" in text):
            findings.append(
                Finding(
                    severity="high",
                    title="可能直接返回 nominal/coarse 轨迹",
                    evidence=str(path),
                    recommendation="确保 planner 输出来自 refined/certified trajectory，而不是 nominal candidate。",
                )
            )
    return findings


def _find_action_agnostic_support(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "compile_support" in text and "action" not in text:
            findings.append(
                Finding(
                    severity="high",
                    title="support compiler 可能不是 action-conditioned",
                    evidence=str(path),
                    recommendation="support 编译入口应显式接收 refined action/path。",
                )
            )
    return findings


def _find_top_mass_only(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"topk.*rho|rho.*topk|sorted\(.*rho", text, re.IGNORECASE):
            if "damage" not in text and "omit" not in text:
                findings.append(
                    Finding(
                        severity="medium",
                        title="可能按 support mass 单独做 top-K",
                        evidence=str(path),
                        recommendation="retention 应优先按 ρ * damage 或等价决策相关性排序。",
                    )
                )
    return findings


def _find_missing_conservative_subset(files: Sequence[Path]) -> list[Finding]:
    conservative_hits = _grep_text(files, r"conservative|fallback|emergency stop")
    if conservative_hits:
        return []
    return [
        Finding(
            severity="high",
            title="未发现 conservative subset / fallback 逻辑",
            evidence="未扫描到 conservative/fallback/emergency stop 关键字",
            recommendation="显式实现 conservative subset、certified fallback 和 emergency stop。",
        )
    ]


def _find_full_cartesian_enumeration(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "itertools.product" in text and "beam" not in text and "prune" not in text:
            findings.append(
                Finding(
                    severity="medium",
                    title="support atom 可能做全组合枚举",
                    evidence=str(path),
                    recommendation="改成 beam search/backtracking，并在每步做 consistency prune。",
                )
            )
    return findings


def _find_geopandas_hot_path(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "geopandas" in text or "gpd.read_file" in text:
            findings.append(
                Finding(
                    severity="medium",
                    title="检测到 GeoPandas 读取路径",
                    evidence=str(path),
                    recommendation="对 nuPlan HD map 元数据提取优先直接读 GeoPackage(SQLite)，避免热路径频繁 `gpd.read_file()`。",
                )
            )
    return findings


def _find_per_sample_sqlite_connect(files: Sequence[Path]) -> list[Finding]:
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "sqlite3.connect" in text:
            findings.append(
                Finding(
                    severity="medium",
                    title="检测到直接 sqlite3.connect",
                    evidence=str(path),
                    recommendation="确保连接是 worker-local 且复用的 read-only connection，不要每样本重复新建。",
                )
            )
    return findings


def run_audit(repo_root: str | Path) -> list[Finding]:
    root = Path(repo_root)
    files = _all_py_files(root)
    findings: list[Finding] = []
    findings.extend(_find_compute_trajectory_returns_nominal(files))
    findings.extend(_find_action_agnostic_support(files))
    findings.extend(_find_top_mass_only(files))
    findings.extend(_find_missing_conservative_subset(files))
    findings.extend(_find_full_cartesian_enumeration(files))
    findings.extend(_find_geopandas_hot_path(files))
    findings.extend(_find_per_sample_sqlite_connect(files))
    return findings


def render_markdown(findings: Sequence[Finding]) -> str:
    if not findings:
        return "# Design Alignment Audit Report\n\n未发现高概率结构性错位模式。\n"
    lines = ["# Design Alignment Audit Report", ""]
    for finding in findings:
        lines.extend(
            [
                f"## [{finding.severity.upper()}] {finding.title}",
                "",
                f"**Evidence**: `{finding.evidence}`",
                "",
                f"**Recommendation**: {finding.recommendation}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an ACS planner repository against implementation-note contracts")
    parser.add_argument("repo_root")
    parser.add_argument("--output", default="design_alignment_report.md")
    args = parser.parse_args()

    findings = run_audit(args.repo_root)
    report = render_markdown(findings)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Wrote audit report to {args.output} with {len(findings)} findings.")


if __name__ == "__main__":
    main()
