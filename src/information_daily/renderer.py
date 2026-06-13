from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import AppConfig, Article, CompiledArticle, CompiledSection, CompiledSubsection, Issue
from .sources import FetchResult


WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

LABELS = {
    "zh": {
        "archive": "往期",
        "archive_title": "往期",
        "articles": "条",
        "back_today": "返回今日",
        "back_overview": "返回总览",
        "for_reading": "仅供阅读",
        "headline": "今日重点",
        "language_zh": "中文",
        "language_en": "English",
        "latest_editions": "本期三版",
        "raw_candidates": "原始候选",
        "source_status": "数据源状态",
        "overview": "总览",
        "overview_rail": "各版速览",
        "overview_footer": "总览 · 三个独立页面",
        "section_pages": "分类页面",
        "subsections": "个细分",
        "read_more": "阅读全文",
        "empty_subsection": "本小版暂无入选文章。你可以在 profile 中调整关键词、数据源或栏目权重。",
        "empty_archive": "暂无归档。",
    },
    "en": {
        "archive": "Archive",
        "archive_title": "Archive",
        "articles": "articles",
        "back_today": "Back to today",
        "back_overview": "Back to overview",
        "for_reading": "For reading only",
        "headline": "Lead item",
        "language_zh": "中文",
        "language_en": "English",
        "latest_editions": "Daily Editions",
        "raw_candidates": "Raw candidates",
        "source_status": "Source status",
        "overview": "Overview",
        "overview_rail": "Edition Briefs",
        "overview_footer": "Overview · Three standalone pages",
        "section_pages": "Edition pages",
        "subsections": "subsections",
        "read_more": "Read more",
        "empty_subsection": "No selected articles in this subsection. Adjust keywords, sources, or weights in the profile.",
        "empty_archive": "No archived issues yet.",
    },
}


def render_issue(
    config: AppConfig,
    issue: Issue,
    out_dir: Path,
    fetch_result: FetchResult | None = None,
    render_raw_pages: bool = True,
) -> None:
    out_dir = out_dir.resolve()
    papers_dir = out_dir / "papers"
    sections_dir = out_dir / "sections"
    en_dir = out_dir / "en"
    en_papers_dir = en_dir / "papers"
    en_sections_dir = en_dir / "sections"
    assets_dir = out_dir / "assets"
    data_dir = config.root / "data" / "issues"
    raw_data_dir = config.root / "data" / "raw"
    papers_dir.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)
    en_papers_dir.mkdir(parents=True, exist_ok=True)
    en_sections_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config.root / "templates" / "assets" / "newspaper.css", assets_dir / "newspaper.css")

    env = _environment(config.root / "templates")
    issue_template = env.get_template("issue.html.j2")
    section_template = env.get_template("section.html.j2")
    archive_template = env.get_template("archive.html.j2")
    raw_href = f"raw/{issue.slug}.html" if fetch_result and render_raw_pages else ""
    sources_href = "sources.html" if fetch_result and render_raw_pages else ""

    context = _issue_context(
        issue,
        config,
        asset_prefix="",
        lang="zh",
        archive_href="archive.html",
        index_href="index.html",
        section_href_prefix="sections/",
        alternate_href="en/index.html",
        raw_href=raw_href,
        sources_href=sources_href,
    )
    (out_dir / "index.html").write_text(issue_template.render(context), encoding="utf-8")
    for section in issue.sections:
        (sections_dir / f"{section.id}.html").write_text(
            section_template.render(
                _section_context(
                    issue,
                    section,
                    config,
                    asset_prefix="../",
                    lang="zh",
                    index_href="../index.html",
                    archive_href="../archive.html",
                    section_href_prefix="",
                    alternate_href=f"../en/sections/{section.id}.html",
                )
            ),
            encoding="utf-8",
        )

    en_context = _issue_context(
        issue,
        config,
        asset_prefix="../",
        lang="en",
        archive_href="archive.html",
        index_href="index.html",
        section_href_prefix="sections/",
        alternate_href="../index.html",
        raw_href=f"../{raw_href}" if raw_href else "",
        sources_href=f"../{sources_href}" if sources_href else "",
    )
    (en_dir / "index.html").write_text(issue_template.render(en_context), encoding="utf-8")
    for section in issue.sections:
        (en_sections_dir / f"{section.id}.html").write_text(
            section_template.render(
                _section_context(
                    issue,
                    section,
                    config,
                    asset_prefix="../../",
                    lang="en",
                    index_href="../index.html",
                    archive_href="../archive.html",
                    section_href_prefix="",
                    alternate_href=f"../../sections/{section.id}.html",
                )
            ),
            encoding="utf-8",
        )

    paper_context = _issue_context(
        issue,
        config,
        asset_prefix="../",
        lang="zh",
        archive_href="../archive.html",
        index_href="../index.html",
        section_href_prefix=f"{issue.slug}/",
        alternate_href=f"../en/papers/{issue.slug}.html",
        raw_href=f"../{raw_href}" if raw_href else "",
        sources_href=f"../{sources_href}" if sources_href else "",
    )
    (papers_dir / f"{issue.slug}.html").write_text(issue_template.render(paper_context), encoding="utf-8")
    dated_sections_dir = papers_dir / issue.slug
    dated_sections_dir.mkdir(parents=True, exist_ok=True)
    for section in issue.sections:
        (dated_sections_dir / f"{section.id}.html").write_text(
            section_template.render(
                _section_context(
                    issue,
                    section,
                    config,
                    asset_prefix="../../",
                    lang="zh",
                    index_href="../../index.html",
                    archive_href="../../archive.html",
                    section_href_prefix="",
                    alternate_href=f"../../en/papers/{issue.slug}/{section.id}.html",
                )
            ),
            encoding="utf-8",
        )

    en_paper_context = _issue_context(
        issue,
        config,
        asset_prefix="../../",
        lang="en",
        archive_href="../archive.html",
        index_href="../index.html",
        section_href_prefix=f"{issue.slug}/",
        alternate_href=f"../../papers/{issue.slug}.html",
        raw_href=f"../../{raw_href}" if raw_href else "",
        sources_href=f"../../{sources_href}" if sources_href else "",
    )
    (en_papers_dir / f"{issue.slug}.html").write_text(issue_template.render(en_paper_context), encoding="utf-8")
    en_dated_sections_dir = en_papers_dir / issue.slug
    en_dated_sections_dir.mkdir(parents=True, exist_ok=True)
    for section in issue.sections:
        (en_dated_sections_dir / f"{section.id}.html").write_text(
            section_template.render(
                _section_context(
                    issue,
                    section,
                    config,
                    asset_prefix="../../../",
                    lang="en",
                    index_href="../../index.html",
                    archive_href="../../archive.html",
                    section_href_prefix="",
                    alternate_href=f"../../../papers/{issue.slug}/{section.id}.html",
                )
            ),
            encoding="utf-8",
        )

    metadata = _issue_metadata(issue)
    (data_dir / f"{issue.slug}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_items = load_archive_items(data_dir, int(config.site.get("archive_limit") or 60))
    (out_dir / "archive.html").write_text(
        archive_template.render(
            site=_localized_site(config, "zh"),
            items=archive_items,
            asset_prefix="",
            index_href="index.html",
            current_href="archive.html",
            alternate_href="en/archive.html",
            lang="zh",
            labels=LABELS["zh"],
            generated_at=_format_datetime(issue.generated_at, config),
        ),
        encoding="utf-8",
    )
    (en_dir / "archive.html").write_text(
        archive_template.render(
            site=_localized_site(config, "en"),
            items=archive_items,
            asset_prefix="../",
            index_href="index.html",
            current_href="archive.html",
            alternate_href="../archive.html",
            lang="en",
            labels=LABELS["en"],
            generated_at=_format_datetime(issue.generated_at, config),
        ),
        encoding="utf-8",
    )

    if fetch_result is not None:
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        raw_snapshot = build_raw_snapshot(config, issue, fetch_result)
        (raw_data_dir / f"{issue.slug}.json").write_text(
            json.dumps(raw_snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if render_raw_pages:
            raw_dir = out_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_template = env.get_template("raw.html.j2")
            sources_template = env.get_template("sources.html.j2")
            source_groups = _raw_source_groups(config, raw_snapshot)
            (raw_dir / f"{issue.slug}.html").write_text(
                raw_template.render(
                    issue=issue,
                    site=config.site,
                    groups=source_groups,
                    generated_at=_format_datetime(issue.generated_at, config),
                    index_href="../index.html",
                    sources_href="../sources.html",
                    archive_href="../archive.html",
                    asset_prefix="../",
                ),
                encoding="utf-8",
            )
            (out_dir / "sources.html").write_text(
                sources_template.render(
                    issue=issue,
                    site=config.site,
                    sources=[group["source"] for group in source_groups],
                    generated_at=_format_datetime(issue.generated_at, config),
                    index_href="index.html",
                    raw_href=raw_href,
                    archive_href="archive.html",
                    asset_prefix="",
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


def _issue_context(
    issue: Issue,
    config: AppConfig,
    asset_prefix: str,
    lang: str,
    archive_href: str,
    index_href: str,
    section_href_prefix: str,
    alternate_href: str,
    raw_href: str = "",
    sources_href: str = "",
) -> dict:
    localized_issue = _localized_issue(issue, lang)
    return {
        "issue": localized_issue,
        "site": _localized_site(config, lang),
        "lang": lang,
        "labels": LABELS[lang],
        "dateline": _format_date(issue.date, lang),
        "generated_at": _format_datetime(issue.generated_at, config),
        "asset_prefix": asset_prefix,
        "archive_href": archive_href,
        "index_href": index_href,
        "current_href": index_href,
        "paper_href_prefix": asset_prefix,
        "section_links": _section_links(localized_issue, section_href_prefix),
        "alternate_href": alternate_href,
        "raw_href": raw_href,
        "sources_href": sources_href,
    }


def _section_context(
    issue: Issue,
    section,
    config: AppConfig,
    asset_prefix: str,
    lang: str,
    index_href: str,
    archive_href: str,
    section_href_prefix: str,
    alternate_href: str,
) -> dict:
    localized_issue = _localized_issue(issue, lang)
    localized_section = next(item for item in localized_issue.sections if item.id == section.id)
    return {
        "issue": localized_issue,
        "section": localized_section,
        "site": _localized_site(config, lang),
        "lang": lang,
        "labels": LABELS[lang],
        "dateline": _format_date(issue.date, lang),
        "generated_at": _format_datetime(issue.generated_at, config),
        "asset_prefix": asset_prefix,
        "archive_href": archive_href,
        "index_href": index_href,
        "current_href": f"{section.id}.html",
        "section_links": _section_links(localized_issue, section_href_prefix, active_section_id=section.id),
        "alternate_href": alternate_href,
    }


def _section_links(issue: Issue, href_prefix: str, active_section_id: str | None = None) -> list[dict]:
    return [
        {
            "section": section,
            "href": f"{href_prefix}{section.id}.html",
            "active": section.id == active_section_id,
        }
        for section in issue.sections
    ]


def _localized_site(config: AppConfig, lang: str) -> dict:
    site = dict(config.site)
    if lang == "en":
        site["title"] = str(config.site.get("title_en") or "Information Daily")
        site["subtitle"] = str(config.site.get("subtitle_en") or config.site.get("subtitle") or "")
        site["description"] = str(config.site.get("description_en") or config.site.get("description") or "")
        site["edition_label"] = str(config.site.get("edition_label_en") or config.site.get("edition_label") or "")
        site["language"] = "en"
    else:
        site["language"] = str(config.site.get("language") or "zh-CN")
    return site


def _localized_issue(issue: Issue, lang: str) -> Issue:
    if lang == "zh":
        return issue

    sections = tuple(_localized_section(section) for section in issue.sections)
    return replace(
        issue,
        site_title=issue.site_title_en or "Information Daily",
        site_subtitle=issue.site_subtitle_en or issue.site_subtitle,
        edition_label=issue.edition_label_en or issue.edition_label,
        briefing_title=issue.briefing_title_en or issue.briefing_title,
        briefing_summary=issue.briefing_summary_en or issue.briefing_summary,
        headline=_localized_article(issue.headline) if issue.headline else None,
        sections=sections,
    )


def _localized_section(section: CompiledSection) -> CompiledSection:
    subsections = tuple(_localized_subsection(subsection) for subsection in section.subsections)
    articles = tuple(_localized_article(article) for article in section.articles)
    return replace(
        section,
        title=section.title_en or section.title,
        briefing_title=section.briefing_title_en or section.briefing_title,
        briefing_summary=section.briefing_summary_en or section.briefing_summary,
        articles=articles,
        subsections=subsections,
    )


def _localized_subsection(subsection: CompiledSubsection) -> CompiledSubsection:
    return replace(
        subsection,
        title=subsection.title_en or subsection.title,
        briefing_title=subsection.briefing_title_en or subsection.briefing_title,
        briefing_summary=subsection.briefing_summary_en or subsection.briefing_summary,
        articles=tuple(_localized_article(article) for article in subsection.articles),
    )


def _localized_article(article: CompiledArticle) -> CompiledArticle:
    return replace(
        article,
        title=article.title_en or article.title,
        summary=article.summary_en or article.summary,
        reason=article.reason_en or article.reason,
    )


def _issue_metadata(issue: Issue) -> dict:
    return {
        "date": issue.slug,
        "title": issue.title,
        "title_en": f"{issue.site_title_en} {issue.slug}",
        "profile_id": issue.profile_id,
        "headline": asdict(issue.headline) if issue.headline else None,
        "path": f"papers/{issue.slug}.html",
        "raw_path": f"raw/{issue.slug}.html",
        "raw_count": issue.raw_count,
        "source_count": issue.source_count,
        "warnings": list(issue.warnings),
    }


def build_raw_snapshot(config: AppConfig, issue: Issue, fetch_result: FetchResult) -> dict:
    return {
        "date": issue.slug,
        "profile_id": config.profile_id,
        "generated_at": issue.generated_at.isoformat() if issue.generated_at else "",
        "sources": [_source_status_payload(status) for status in fetch_result.sources],
        "articles": [
            _article_payload(article, rank)
            for rank, article in enumerate(fetch_result.articles, start=1)
        ],
        "warnings": list(issue.warnings),
    }


def _source_status_payload(status: object) -> dict:
    return asdict(status)


def _article_payload(article: Article, rank: int) -> dict:
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source_id": article.source_id,
        "source_name": article.source_name,
        "default_section": article.default_section,
        "summary": article.summary,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "weight": article.weight,
        "rank": rank,
    }


def _raw_source_groups(config: AppConfig, raw_snapshot: dict) -> list[dict]:
    display_by_id = {source.id: source.display for source in config.sources}
    articles_by_source: dict[str, list[dict]] = {}
    for article in raw_snapshot["articles"]:
        articles_by_source.setdefault(article["source_id"], []).append(article)

    groups = []
    for source in raw_snapshot["sources"]:
        if not display_by_id.get(source["id"], True):
            continue
        decorated = {
            **source,
            "status_label": _status_label(str(source.get("status") or "")),
        }
        groups.append(
            {
                "source": decorated,
                "articles": articles_by_source.get(source["id"], []),
            }
        )
    return groups


def _status_label(status: str) -> str:
    if status == "success":
        return "成功"
    if status == "disabled":
        return "停用"
    return "失败"


def _format_date(value: date, lang: str = "zh") -> str:
    if lang == "en":
        return f"{value.strftime('%B')} {value.day}, {value.year} · {WEEKDAYS_EN[value.weekday()]}"
    return f"{value.year}年{value.month}月{value.day}日 {WEEKDAYS[value.weekday()]}"


def _format_datetime(value: datetime | None, config: AppConfig) -> str:
    if value is None:
        return ""
    timezone = ZoneInfo(str(config.site.get("timezone") or "UTC"))
    local = value.astimezone(timezone)
    return local.strftime("%Y-%m-%d %H:%M %Z")
