from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from .compiler import compile_issue, issue_to_dict
from .config import ConfigError, load_config, validate_config
from .renderer import render_issue
from .models import Article
from .sources import FetchResult, SourceFetchStatus, fetch_all


def _load_source_snapshot(path: Path, profile: str, issue_date: date) -> FetchResult:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot["profile_id"] != profile or snapshot["date"] != issue_date.isoformat():
        raise ValueError("Source snapshot profile/date differs from requested edition")
    return FetchResult(
        articles=tuple(
            Article(
                **{
                    k: v
                    for k, v in row.items()
                    if k in Article.__dataclass_fields__ and k != "published_at"
                },
                published_at=datetime.fromisoformat(row["published_at"])
                if row.get("published_at")
                else None,
            )
            for row in snapshot["articles"]
        ),
        # Legacy snapshots mix fetch errors with the previous edition's editor
        # warnings. Replaying sources must not carry old rejection counts forward.
        warnings=tuple(
            snapshot.get(
                "fetch_warnings",
                [
                    row["error"]
                    for row in snapshot["sources"]
                    if row.get("status") == "error" and row.get("error")
                ],
            )
        ),
        sources=tuple(SourceFetchStatus(**row) for row in snapshot["sources"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="information-daily")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config", help="Validate site/profile/source config"
    )
    validate.add_argument("--profile", default="ai-tech")
    validate.add_argument("--root", default=".")

    generate = subparsers.add_parser("generate", help="Generate static newspaper pages")
    generate.add_argument("--profile", default="ai-tech")
    generate.add_argument("--date", default=date.today().isoformat())
    generate.add_argument("--out", default="docs")
    generate.add_argument("--root", default=".")
    generate.add_argument("--source-timeout", type=int, default=12)
    generate.add_argument("--fetch-workers", type=int, default=8)
    generate.add_argument(
        "--issue-json", type=Path, help="Save the full compiled edition for review."
    )
    generate.add_argument(
        "--source-snapshot",
        type=Path,
        help="Reuse a saved RSS snapshot instead of fetching new sources.",
    )
    generate.add_argument(
        "--editorial-audit",
        type=Path,
        help="Save raw editor selections and every validation decision, including on failure.",
    )
    generate.add_argument(
        "--editorial-selection",
        type=Path,
        help="Reuse saved API article selections; requires a matching source snapshot. Topic review and summaries still call the API.",
    )
    generate.add_argument(
        "--skip-raw-pages",
        action="store_true",
        help="Write the raw JSON snapshot but skip raw/source browsing pages.",
    )
    generate.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Use rule-based compilation if LLM credentials/request are unavailable. Do not use in production.",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = validate_config(profile=args.profile, root=Path(args.root))
            print(
                f"ok: profile={config.profile_id}, sections={len(config.sections)}, sources={len(config.sources)}"
            )
            return 0
        if args.command == "generate":
            config = load_config(profile=args.profile, root=Path(args.root))
            issue_date = date.fromisoformat(args.date)
            fetch = (
                _load_source_snapshot(
                    args.source_snapshot, config.profile_id, issue_date
                )
                if args.source_snapshot
                else fetch_all(
                    config, timeout=args.source_timeout, workers=args.fetch_workers
                )
            )
            audit = {} if args.editorial_audit else None
            selection = None
            if args.editorial_selection:
                if not args.source_snapshot:
                    raise ValueError(
                        "Saved editorial selections require a source snapshot"
                    )
                selection = json.loads(
                    args.editorial_selection.read_text(encoding="utf-8")
                )
                if selection.get("date") != issue_date.isoformat():
                    raise ValueError("Saved editorial selection date differs")
            try:
                issue = compile_issue(
                    config,
                    fetch.articles,
                    issue_date=issue_date,
                    allow_fallback=args.allow_fallback,
                    editorial_audit=audit,
                    editorial_selection=selection,
                )
            finally:
                if args.editorial_audit:
                    args.editorial_audit.parent.mkdir(parents=True, exist_ok=True)
                    args.editorial_audit.write_text(
                        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            if fetch.warnings:
                issue = issue.__class__(
                    **{
                        field: getattr(issue, field)
                        for field in issue.__dataclass_fields__
                        if field != "warnings"
                    },
                    warnings=tuple([*issue.warnings, *fetch.warnings]),
                )
            render_issue(
                config,
                issue,
                Path(args.out),
                fetch_result=fetch,
                render_raw_pages=not args.skip_raw_pages,
            )
            if args.issue_json:
                args.issue_json.parent.mkdir(parents=True, exist_ok=True)
                args.issue_json.write_text(
                    json.dumps(
                        issue_to_dict(issue),
                        ensure_ascii=False,
                        indent=2,
                        default=lambda value: value.isoformat(),
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(f"generated: {Path(args.out).resolve() / 'index.html'}")
            return 0
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    return 1
