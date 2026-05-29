from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

from .config import ConfigError
from .models import AppConfig, Article, CompiledArticle, CompiledSection, Issue, SectionConfig
from .utils import normalize_space


class CompileError(RuntimeError):
    """Raised when the digest cannot be compiled."""


def compile_issue(
    config: AppConfig,
    articles: list[Article] | tuple[Article, ...],
    issue_date: date,
    allow_fallback: bool = False,
) -> Issue:
    prepared = _prepare_articles(config, articles, issue_date)
    warnings: list[str] = []

    if config.llm.enabled:
        try:
            issue = _compile_with_llm(config, prepared, issue_date)
            return issue
        except ConfigError:
            if not allow_fallback:
                raise
            warnings.append("LLM credentials are missing; generated with rule-based fallback.")
        except CompileError:
            if not allow_fallback:
                raise
            warnings.append("LLM compilation failed; generated with rule-based fallback.")

    issue = _compile_with_rules(config, prepared, issue_date)
    return Issue(
        **{field: getattr(issue, field) for field in issue.__dataclass_fields__ if field != "warnings"},
        warnings=tuple([*issue.warnings, *warnings]),
    )


def _prepare_articles(
    config: AppConfig,
    articles: list[Article] | tuple[Article, ...],
    issue_date: date,
) -> list[Article]:
    del issue_date
    seen: set[str] = set()
    deduped: list[Article] = []
    for article in articles:
        title_key = normalize_space(article.title).lower()
        key = article.url or title_key
        if key in seen or title_key in seen:
            continue
        seen.add(key)
        seen.add(title_key)
        deduped.append(article)
    return sorted(
        deduped,
        key=lambda item: (
            item.published_at is not None,
            item.published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
            item.weight,
        ),
        reverse=True,
    )[: int(config.selection.get("max_raw_items") or 120)]


def _compile_with_rules(config: AppConfig, articles: list[Article], issue_date: date) -> Issue:
    scored: list[tuple[Article, SectionConfig, float]] = []
    for article in articles:
        section, score = _best_section(config, article)
        if score >= float(config.selection.get("min_score") or 0):
            scored.append((article, section, score))
    scored.sort(key=lambda row: (row[2], row[0].published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC"))), reverse=True)

    headline = _compiled_article(scored[0][0], scored[0][2]) if scored else None
    sections: list[CompiledSection] = []
    used_headline_url = headline.url if headline else None
    for section in config.sections:
        items = [
            (article, score)
            for article, item_section, score in scored
            if item_section.id == section.id and article.url != used_headline_url
        ][: section.max_articles]
        sections.append(
            CompiledSection(
                id=section.id,
                title=section.title,
                icon=section.icon,
                articles=tuple(_compiled_article(article, score) for article, score in items),
            )
        )

    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_subtitle=str(config.site.get("subtitle") or ""),
        edition_label=str(config.site.get("edition_label") or ""),
        profile_id=config.profile_id,
        briefing_title="AI 总结",
        briefing_summary=_fallback_briefing(scored),
        headline=headline,
        sections=tuple(sections),
        source_count=len([source for source in config.sources if source.enabled]),
        raw_count=len(articles),
        warnings=(),
        generated_at=_now(config),
    )


def _compile_with_llm(config: AppConfig, articles: list[Article], issue_date: date) -> Issue:
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise ConfigError(f"Missing required LLM secret: {config.llm.api_key_env}")

    candidates = []
    for article in articles[: config.llm.max_input_items]:
        section, score = _best_section(config, article)
        candidates.append(
            {
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "source": article.source_name,
                "summary": article.summary[:700],
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "suggested_section": section.id,
                "score": round(score, 2),
            }
        )

    payload = {
        "model": os.environ.get(config.llm.model_env, config.llm.default_model),
        "temperature": config.llm.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是严格的中文日报编辑和翻译。只能基于输入候选文章编纂，不得编造事实、链接、来源或日期。"
                    "所有新闻标题、摘要和入选理由必须用简体中文表达；英文只允许保留公司名、产品名、论文名、专有名词和来源名。"
                    "如果原文标题是英文，也要翻译成自然的中文新闻标题。输出必须是合法 JSON，不要 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "从候选文章中生成一份中文个人日报。",
                        "date": issue_date.isoformat(),
                        "style": config.llm.summary_style,
                        "sections": [
                            {
                                "id": section.id,
                                "title": section.title,
                                "icon": section.icon,
                                "max_articles": section.max_articles,
                            }
                            for section in config.sections
                        ],
                        "schema": {
                            "briefing": {
                                "title": "例如：AI 总结",
                                "summary": "180-320字简体中文，概括今天最重要的趋势、变化和阅读重点",
                            },
                            "headline": {
                                "title": "简体中文新闻标题",
                                "url": "string",
                                "source": "string",
                                "summary": "80-180字中文摘要",
                                "reason": "为什么作为头条",
                                "score": "number",
                            },
                            "sections": [
                                {
                                    "id": "section id",
                                    "articles": [
                                        {
                                            "title": "简体中文新闻标题",
                                            "url": "string",
                                            "source": "string",
                                            "summary": "60-140字中文摘要",
                                            "reason": "入选理由",
                                            "score": "number",
                                        }
                                    ],
                                }
                            ],
                        },
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    raw = _post_chat_completion(config, payload, api_key)
    data = _extract_json(raw)
    return _issue_from_llm_data(config, data, articles, issue_date)


def _post_chat_completion(config: AppConfig, payload: dict, api_key: str) -> str:
    base_url = os.environ.get(config.llm.base_url_env, config.llm.default_base_url).rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CompileError(f"LLM request failed: {exc}") from exc
    try:
        return response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CompileError("LLM response did not include choices[0].message.content") from exc


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise CompileError("LLM response did not contain a JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CompileError("LLM response JSON could not be parsed") from exc
    if not isinstance(data, dict):
        raise CompileError("LLM response JSON must be an object")
    return data


def _issue_from_llm_data(
    config: AppConfig,
    data: dict,
    articles: list[Article],
    issue_date: date,
) -> Issue:
    allowed_by_url = {article.url: article for article in articles}
    headline = _compiled_from_mapping(data.get("headline"), allowed_by_url)
    if headline is None and articles:
        headline = _compiled_article(articles[0], 0)
    briefing = data.get("briefing") if isinstance(data.get("briefing"), dict) else {}

    llm_sections_by_id = {}
    for item in data.get("sections") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            llm_sections_by_id[item["id"]] = item

    sections: list[CompiledSection] = []
    for section in config.sections:
        raw_items = (llm_sections_by_id.get(section.id) or {}).get("articles") or []
        compiled = tuple(
            item
            for item in (_compiled_from_mapping(raw_item, allowed_by_url) for raw_item in raw_items)
            if item is not None
        )[: section.max_articles]
        sections.append(
            CompiledSection(
                id=section.id,
                title=section.title,
                icon=section.icon,
                articles=compiled,
            )
        )

    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_subtitle=str(config.site.get("subtitle") or ""),
        edition_label=str(config.site.get("edition_label") or ""),
        profile_id=config.profile_id,
        briefing_title=normalize_space(str(briefing.get("title") or "AI 总结")),
        briefing_summary=normalize_space(str(briefing.get("summary") or _fallback_briefing([]))),
        headline=headline,
        sections=tuple(sections),
        source_count=len([source for source in config.sources if source.enabled]),
        raw_count=len(articles),
        warnings=(),
        generated_at=_now(config),
    )


def _compiled_from_mapping(value: object, allowed_by_url: dict[str, Article]) -> CompiledArticle | None:
    if not isinstance(value, dict):
        return None
    url = str(value.get("url") or "")
    source_article = allowed_by_url.get(url)
    if source_article is None:
        return None
    return CompiledArticle(
        title=normalize_space(str(value.get("title") or source_article.title)),
        url=url,
        source=normalize_space(str(value.get("source") or source_article.source_name)),
        summary=normalize_space(str(value.get("summary") or source_article.summary)),
        published_at=source_article.published_at.isoformat() if source_article.published_at else None,
        reason=normalize_space(str(value.get("reason") or "")),
        score=float(value.get("score") or 0),
    )


def _best_section(config: AppConfig, article: Article) -> tuple[SectionConfig, float]:
    text = f"{article.title} {article.summary} {' '.join(article.keywords)}".lower()
    best = config.sections[0]
    best_score = -1.0
    for section in config.sections:
        score = article.weight
        if article.default_section == section.id:
            score += 2.0
        score += sum(1.0 for keyword in section.keywords if keyword and keyword in text)
        if score > best_score:
            best = section
            best_score = score
    return best, best_score


def _compiled_article(article: Article, score: float) -> CompiledArticle:
    summary = article.summary or article.title
    if len(summary) > 220:
        summary = summary[:217].rstrip() + "..."
    return CompiledArticle(
        title=article.title,
        url=article.url,
        source=article.source_name,
        summary=summary,
        published_at=article.published_at.isoformat() if article.published_at else None,
        score=round(score, 2),
    )


def _fallback_briefing(scored: list[tuple[Article, SectionConfig, float]]) -> str:
    if not scored:
        return "今日暂未获取到足够新闻。请检查数据源、关键词或 LLM 配置后重新生成。"
    top_sources = []
    for article, _, _ in scored[:5]:
        if article.source_name not in top_sources:
            top_sources.append(article.source_name)
    return (
        "这是本地规则模式生成的版式预览，未进行正式中文翻译。"
        "生产环境配置 LLM 后，这里会自动汇总今日重点、翻译标题与摘要，并按栏目生成中文日报。"
        f" 本次预览共读取 {len(scored)} 条候选内容，主要来源包括：{'、'.join(top_sources)}。"
    )


def _now(config: AppConfig) -> datetime:
    timezone_name = str(config.site.get("timezone") or "UTC")
    return datetime.now(ZoneInfo(timezone_name))


def issue_to_dict(issue: Issue) -> dict:
    return asdict(issue)
