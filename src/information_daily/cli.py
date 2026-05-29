from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .compiler import compile_issue
from .config import ConfigError, load_config, validate_config
from .renderer import render_issue
from .sources import fetch_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="information-daily")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate site/profile/source config")
    validate.add_argument("--profile", default="ai-tech")
    validate.add_argument("--root", default=".")

    generate = subparsers.add_parser("generate", help="Generate static newspaper pages")
    generate.add_argument("--profile", default="ai-tech")
    generate.add_argument("--date", default=date.today().isoformat())
    generate.add_argument("--out", default="docs")
    generate.add_argument("--root", default=".")
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
            fetch = fetch_all(config)
            issue = compile_issue(
                config,
                fetch.articles,
                issue_date=issue_date,
                allow_fallback=args.allow_fallback,
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
            render_issue(config, issue, Path(args.out))
            print(f"generated: {Path(args.out).resolve() / 'index.html'}")
            return 0
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    return 1
