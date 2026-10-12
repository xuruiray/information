"""Render a Codex-authored editorial review through the production validators.

This consumes saved source material and an explicit editorial JSON artifact. It
makes no API calls and is not evidence that the live model transport works.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from information_daily.compiler import (
    CompileError,
    _apply_final_briefings,
    _apply_final_overview,
    _issue_from_llm_data,
    _prepare_articles,
)
from information_daily.config import load_config
from information_daily.models import Article
from information_daily.renderer import render_issue
from information_daily.sources import FetchResult, SourceFetchStatus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--editorial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", default="ai-tech")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    editorial = json.loads(args.editorial.read_text())
    issue_date = date.fromisoformat(snapshot["date"])
    if editorial["provenance"]["date"] != snapshot["date"]:
        raise ValueError("Editorial and snapshot dates differ")
    if editorial["provenance"]["mode"] != "offline-review":
        raise ValueError("This command only renders explicit offline reviews")
    articles = tuple(
        Article(
            **{
                key: value
                for key, value in row.items()
                if key in Article.__dataclass_fields__ and key != "published_at"
            },
            published_at=datetime.fromisoformat(row["published_at"])
            if row.get("published_at")
            else None,
        )
        for row in snapshot["articles"]
    )
    out = args.out.resolve()
    with TemporaryDirectory(prefix="information-review-") as workspace:
        root = Path(workspace)
        for folder in ("config", "templates"):
            copytree(args.root / folder, root / folder)
        config = load_config(args.profile, root)
        config = replace(
            config,
            site={
                **config.site,
                "edition_label": "Codex 编纂审阅版",
                "edition_label_en": "Codex editorial review",
            },
        )
        issue = _issue_from_llm_data(
            config,
            editorial,
            _prepare_articles(config, articles, issue_date),
            issue_date,
        )
        if issue.warnings:
            raise CompileError(
                "Review selection failed validation: " + "; ".join(issue.warnings)
            )
        summaries = {entry["id"]: entry for entry in editorial["final_briefings"]}
        if set(summaries) != {s.id for s in issue.sections if s.articles}:
            raise CompileError("Review summary sections differ from final selection")
        issue = replace(
            issue,
            sections=tuple(
                _apply_final_briefings(section, summaries[section.id])
                if section.articles
                else section
                for section in issue.sections
            ),
        )
        issue = _apply_final_overview(issue, editorial["overview"])
        fetch = FetchResult(
            articles=articles,
            warnings=tuple(snapshot.get("warnings", [])),
            sources=tuple(SourceFetchStatus(**row) for row in snapshot["sources"]),
        )
        render_issue(config, issue, out, fetch_result=fetch)
        (out / "edition.json").write_text(
            json.dumps(
                {"provenance": editorial["provenance"], "issue": asdict(issue)},
                ensure_ascii=False,
                indent=2,
                default=lambda value: value.isoformat(),
            )
            + "\n"
        )
        print(f"Generated offline review: {out / 'index.html'}")
        print(
            f"Articles: {sum(len(s.articles) for s in issue.sections)}; warnings: {len(issue.warnings)}"
        )


if __name__ == "__main__":
    main()
