from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import AppConfig, Issue


WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def render_issue(config: AppConfig, issue: Issue, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    papers_dir = out_dir / "papers"
    assets_dir = out_dir / "assets"
    data_dir = config.root / "data" / "issues"
    papers_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config.root / "templates" / "assets" / "newspaper.css", assets_dir / "newspaper.css")

    env = _environment(config.root / "templates")
    issue_template = env.get_template("issue.html.j2")
    archive_template = env.get_template("archive.html.j2")

    context = _issue_context(issue, config, asset_prefix="")
    (out_dir / "index.html").write_text(issue_template.render(context), encoding="utf-8")

    paper_context = _issue_context(issue, config, asset_prefix="../")
    (papers_dir / f"{issue.slug}.html").write_text(issue_template.render(paper_context), encoding="utf-8")

    metadata = _issue_metadata(issue)
    (data_dir / f"{issue.slug}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_items = load_archive_items(data_dir, int(config.site.get("archive_limit") or 60))
    (out_dir / "archive.html").write_text(
        archive_template.render(
            site=config.site,
            items=archive_items,
            asset_prefix="",
            generated_at=_format_datetime(issue.generated_at, config),
        ),
        encoding="utf-8",
    )


def load_archive_items(data_dir: Path, limit: int) -> list[dict]:
    items: list[dict] = []
    if not data_dir.exists():
        return items
    for path in sorted(data_dir.glob("*.json"), reverse=True):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return items[:limit]


def _environment(template_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(("html", "xml", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def _issue_context(issue: Issue, config: AppConfig, asset_prefix: str) -> dict:
    return {
        "issue": issue,
        "site": config.site,
        "dateline": _format_date(issue.date),
        "generated_at": _format_datetime(issue.generated_at, config),
        "asset_prefix": asset_prefix,
        "archive_href": f"{asset_prefix}archive.html",
        "index_href": f"{asset_prefix}index.html",
        "paper_href_prefix": asset_prefix,
    }


def _issue_metadata(issue: Issue) -> dict:
    return {
        "date": issue.slug,
        "title": issue.title,
        "profile_id": issue.profile_id,
        "headline": asdict(issue.headline) if issue.headline else None,
        "path": f"papers/{issue.slug}.html",
        "raw_count": issue.raw_count,
        "source_count": issue.source_count,
    }


def _format_date(value: date) -> str:
    return f"{value.year}年{value.month}月{value.day}日 {WEEKDAYS[value.weekday()]}"


def _format_datetime(value: datetime | None, config: AppConfig) -> str:
    if value is None:
        return ""
    timezone = ZoneInfo(str(config.site.get("timezone") or "UTC"))
    local = value.astimezone(timezone)
    return local.strftime("%Y-%m-%d %H:%M %Z")
