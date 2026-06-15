from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .config import ConfigError
from .models import (
    AppConfig,
    Article,
    CompiledArticle,
    CompiledSection,
    CompiledSubsection,
    Issue,
    SectionConfig,
    SubsectionConfig,
)
from .utils import normalize_space


class CompileError(RuntimeError):
    """Raised when the digest cannot be compiled."""


MIN_SUBSECTION_ARTICLES = 3


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
    timezone = ZoneInfo(str(config.site.get("timezone") or "UTC"))
    cutoff = None
    lookback_hours = config.selection.get("lookback_hours")
    if lookback_hours:
        issue_end = datetime.combine(issue_date + timedelta(days=1), time.min, tzinfo=timezone)
        cutoff = issue_end - timedelta(hours=float(lookback_hours))

    seen: set[str] = set()
    deduped: list[Article] = []
    for article in articles:
        if cutoff and article.published_at and article.published_at.astimezone(timezone) < cutoff:
            continue
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
    used_urls = {_article_key_from_compiled(headline)} if headline else set()
    for section in config.sections:
        subsection_items = _balanced_subsection_pairs(section, scored, used_urls)
        items = [
            pair
            for subsection in _section_subsections(section)
            for pair in subsection_items.get(subsection.id, [])
        ]
        used_urls.update(_article_key(article) for article, _ in items)
        subsections = _compiled_subsections_from_grouped(section, subsection_items)
        sections.append(
            CompiledSection(
                id=section.id,
                title=section.title,
                title_en=section.title_en,
                icon=section.icon,
                articles=tuple(_compiled_article(article, score) for article, score in items),
                briefing_title=f"{section.title}总结",
                briefing_summary=_fallback_section_briefing(section, items),
                briefing_title_en=f"{section.title_en} Briefing",
                briefing_summary_en=_fallback_section_briefing_en(section, items),
                subsections=subsections,
            )
        )

    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_title_en=str(config.site.get("title_en") or "Information Daily"),
        site_subtitle=str(config.site.get("subtitle") or ""),
        site_subtitle_en=str(config.site.get("subtitle_en") or config.site.get("subtitle") or ""),
        edition_label=str(config.site.get("edition_label") or ""),
        edition_label_en=str(config.site.get("edition_label_en") or config.site.get("edition_label") or ""),
        profile_id=config.profile_id,
        briefing_title="AI 总结",
        briefing_summary=_fallback_briefing(scored),
        briefing_title_en="AI Briefing",
        briefing_summary_en=_fallback_briefing_en(scored),
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

    ranked_by_section = _ranked_candidates_by_section(config, articles)
    ranked_by_subsection = _ranked_candidates_by_subsection(config, articles)
    overview = _compile_llm_overview(config, ranked_by_section, issue_date, api_key)
    section_data = [
        _compile_llm_section(config, section, ranked_by_section, ranked_by_subsection, issue_date, api_key)
        for section in config.sections
    ]

    return _issue_from_llm_data(
        config,
        {
            "briefing": overview.get("briefing") or {},
            "headline": overview.get("headline") or {},
            "sections": section_data,
        },
        articles,
        issue_date,
    )


def _compile_llm_overview(
    config: AppConfig,
    ranked_by_section: dict[str, list[tuple[Article, float]]],
    issue_date: date,
    api_key: str,
) -> dict:
    llm_candidates_by_section = _llm_candidates_by_section(config, ranked_by_section)
    candidates_by_section = {}
    for section in config.sections:
        section_candidates = llm_candidates_by_section.get(section.id, [])
        candidates_by_section[section.id] = [
            _article_candidate_payload(article, score, section)
            for article, score in section_candidates[: min(8, max(section.max_articles, 1))]
        ]

    payload = _llm_payload(
        config,
        [
            {"role": "system", "content": _llm_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "生成整份日报的总览和头条。",
                        "date": issue_date.isoformat(),
                        "style": config.llm.summary_style,
                        "sections": [
                            {
                                "id": section.id,
                                "title": section.title,
                                "title_en": section.title_en,
                            }
                            for section in config.sections
                        ],
                        "requirements": [
                            "只输出 briefing 和 headline 两个字段。",
                            "headline 必须从 candidates_by_section 中选择，url 必须完全一致。",
                            "briefing.summary 要综合所有配置版面，突出当天最重要的趋势、风险和阅读优先级。",
                            "中文字段必须是自然简体中文，不要直接复述英文原文。",
                        ],
                        "schema": {
                            "briefing": {
                                "title": "例如：AI 总结",
                                "summary": "160-260字简体中文，概括今天最重要的趋势、变化和阅读重点",
                                "title_en": "example: AI Briefing",
                                "summary_en": "120-220 words in English, covering the most important trends and reading priorities",
                            },
                            "headline": {
                                "title": "简体中文新闻标题",
                                "title_en": "English news title",
                                "url": "string",
                                "source_zh": "中文来源名",
                                "source_en": "English source name",
                                "summary": "70-140字中文摘要",
                                "summary_en": "60-120 word English summary",
                                "reason": "为什么作为头条",
                                "reason_en": "why this is the lead item",
                                "score": "number",
                            },
                        },
                        "candidates_by_section": candidates_by_section,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return _extract_json(_post_chat_completion(config, payload, api_key))


def _compile_llm_section(
    config: AppConfig,
    section: SectionConfig,
    ranked_by_section: dict[str, list[tuple[Article, float]]],
    ranked_by_subsection: dict[str, dict[str, list[tuple[Article, float]]]],
    issue_date: date,
    api_key: str,
) -> dict:
    targets_by_section = {
        section.id: min(section.max_articles, len(ranked_by_section.get(section.id, [])))
    }
    targets_by_subsection = {
        subsection.id: min(
            subsection.max_articles,
            len(ranked_by_subsection.get(section.id, {}).get(subsection.id, [])),
        )
        for subsection in _section_subsections(section)
    }
    section_candidates = ranked_by_section.get(section.id, [])
    section_candidate_limit = min(
        max(1, int(config.llm.max_input_items or 60)),
        max(section.max_articles * 2, section.max_articles),
    )
    candidates = [
        _article_candidate_payload(article, score, section)
        for article, score in section_candidates[:section_candidate_limit]
    ]

    payload = _llm_payload(
        config,
        [
            {"role": "system", "content": _llm_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": f"生成 {section.title} 大版的分版总结、小版块总结和文章列表。",
                        "date": issue_date.isoformat(),
                        "style": config.llm.summary_style,
                        "section": {
                            "id": section.id,
                            "title": section.title,
                            "title_en": section.title_en,
                            "icon": section.icon,
                            "max_articles": section.max_articles,
                            "target_articles": targets_by_section[section.id],
                            "subsections": [
                                {
                                    "id": subsection.id,
                                    "title": subsection.title,
                                    "title_en": subsection.title_en,
                                    "max_articles": subsection.max_articles,
                                    "target_articles": targets_by_subsection[subsection.id],
                                }
                                for subsection in _section_subsections(section)
                            ],
                        },
                        "requirements": [
                            "顶层必须是一个 section 对象，id 必须等于输入 section.id。",
                            "subsections 必须覆盖输入 section.subsections 的所有 id。",
                            "每个 subsections[j].articles 的数量必须尽量等于该小版 target_articles；候选不足时可以少于 target_articles。",
                            "所有文章只能从 candidates 中选择，url 必须完全一致。",
                            "小版块优先使用 candidate.suggested_subsection 匹配的文章；候选不足时可从同一大版其他候选补齐。",
                            "中文标题、摘要和入选理由必须是正式读者可读的新闻文案，不要出现 prompt、规则预览、LLM、生产环境等内部说明。",
                        ],
                        "schema": {
                            "id": section.id,
                            "briefing": {
                                "title": f"例如：{section.title}总结",
                                "summary": "100-180字简体中文，概括本版新闻重点",
                                "title_en": f"example: {section.title_en} Briefing",
                                "summary_en": "80-140 word English summary of this edition",
                            },
                            "subsections": [
                                {
                                    "id": "subsection id",
                                    "briefing": {
                                        "title": "例如：小版总结",
                                        "summary": "60-120字简体中文，概括该小版块重点",
                                        "title_en": "example: Subsection Briefing",
                                        "summary_en": "50-100 word English summary of this subsection",
                                    },
                                    "articles": [
                                        {
                                            "title": "简体中文新闻标题",
                                            "title_en": "English news title",
                                            "url": "string",
                                            "source_zh": "中文来源名",
                                            "source_en": "English source name",
                                            "summary": "50-110字中文摘要",
                                            "summary_en": "45-90 word English summary",
                                            "reason": "入选理由",
                                            "reason_en": "selection rationale",
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
    )
    raw = _post_chat_completion(config, payload, api_key)
    data = _extract_json(raw)
    return _coerce_llm_section_data(section, data)


def _llm_payload(config: AppConfig, messages: list[dict]) -> dict:
    payload = {
        "model": os.environ.get(config.llm.model_env, config.llm.default_model),
        "temperature": config.llm.temperature,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    max_tokens = os.environ.get("OPENAI_MAX_TOKENS")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    return payload


def _llm_system_prompt() -> str:
    return (
        "你是严格的中文日报编辑和翻译。只能基于输入候选文章编纂，不得编造事实、链接、来源或日期。"
        "所有新闻标题、摘要和入选理由必须同时输出简体中文字段和英文字段。"
        "中文字段使用自然简体中文；英文字段使用清晰新闻英语。"
        "如果原文标题是英文，也要翻译成自然的中文新闻标题，同时保留准确英文标题。"
        "输出必须是合法 JSON，不要 Markdown，不要解释。"
    )


def _coerce_llm_section_data(section: SectionConfig, data: dict) -> dict:
    if data.get("id") == section.id:
        return data
    raw_section = data.get("section")
    if isinstance(raw_section, dict) and raw_section.get("id") == section.id:
        return raw_section
    for raw_section in data.get("sections") or []:
        if isinstance(raw_section, dict) and raw_section.get("id") == section.id:
            return raw_section
    data["id"] = section.id
    return data


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
    allowed_by_url = {article.url: article for article in articles if article.url}
    ranked_by_section = _ranked_candidates_by_section(config, articles)
    ranked_by_subsection = _ranked_candidates_by_subsection(config, articles)
    warnings: list[str] = []
    headline, warning = _compiled_from_mapping_checked(data.get("headline"), allowed_by_url)
    if warning and data.get("headline") is not None:
        warnings.append(f"LLM headline ignored: {warning}")
    if headline is None and articles:
        headline = _compiled_article(articles[0], 0)
    briefing = data.get("briefing") if isinstance(data.get("briefing"), dict) else {}

    llm_sections_by_id = {}
    for item in data.get("sections") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            llm_sections_by_id[item["id"]] = item

    sections: list[CompiledSection] = []
    used_urls = {headline.url} if headline else set()
    for section in config.sections:
        raw_section = llm_sections_by_id.get(section.id) or {}
        if not raw_section:
            warnings.append(f"LLM output missed section {section.id}; filled with ranked candidates.")
        raw_briefing = raw_section.get("briefing") if isinstance(raw_section.get("briefing"), dict) else {}
        raw_subsections_by_id = {
            item["id"]: item
            for item in raw_section.get("subsections") or []
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        subsection_results: list[CompiledSubsection] = []
        section_compiled: list[CompiledArticle] = []
        for subsection in _section_subsections(section):
            raw_subsection = raw_subsections_by_id.get(subsection.id) or {}
            if not raw_subsection:
                warnings.append(
                    f"LLM output missed subsection {section.id}/{subsection.id}; filled with ranked candidates."
                )
            raw_items = raw_subsection.get("articles") or []
            if not isinstance(raw_items, list):
                warnings.append(
                    f"LLM subsection {section.id}/{subsection.id} articles was not a list; filled with ranked candidates."
                )
                raw_items = []
            raw_subsection_briefing = (
                raw_subsection.get("briefing") if isinstance(raw_subsection.get("briefing"), dict) else {}
            )
            compiled: list[CompiledArticle] = []
            for raw_item in raw_items:
                item, warning = _compiled_from_mapping_checked(raw_item, allowed_by_url)
                if warning:
                    warnings.append(
                        f"LLM subsection {section.id}/{subsection.id} ignored item: {warning}"
                    )
                    continue
                if item is None:
                    continue
                if item.url in used_urls:
                    warnings.append(
                        f"LLM subsection {section.id}/{subsection.id} ignored duplicate URL: {item.url}"
                    )
                    continue
                compiled.append(item)
                used_urls.add(item.url)
                if len(compiled) >= subsection.max_articles:
                    break
            for article, score in ranked_by_subsection.get(section.id, {}).get(subsection.id, []):
                if len(compiled) >= subsection.max_articles or len(section_compiled) + len(compiled) >= section.max_articles:
                    break
                if article.url in used_urls:
                    continue
                compiled.append(_compiled_article(article, score))
                used_urls.add(article.url)
            subsection_results.append(
                CompiledSubsection(
                    id=subsection.id,
                    title=subsection.title,
                    title_en=subsection.title_en,
                    articles=tuple(compiled),
                    briefing_title=normalize_space(
                        str(raw_subsection_briefing.get("title") or f"{subsection.title}总结")
                    ),
                    briefing_summary=normalize_space(
                        str(raw_subsection_briefing.get("summary") or _fallback_subsection_briefing(subsection, []))
                    ),
                    briefing_title_en=normalize_space(
                        str(raw_subsection_briefing.get("title_en") or f"{subsection.title_en} Briefing")
                    ),
                    briefing_summary_en=normalize_space(
                        str(
                            raw_subsection_briefing.get("summary_en")
                            or _fallback_subsection_briefing_en(subsection, [])
                        )
                    ),
                )
            )
            section_compiled.extend(compiled)

        for article, score in ranked_by_section.get(section.id, []):
            if len(section_compiled) >= section.max_articles:
                break
            if article.url in used_urls:
                continue
            compiled = _compiled_article(article, score)
            section_compiled.append(compiled)
            used_urls.add(article.url)
            subsection_id = _best_subsection(section, article).id
            subsection_results = tuple(
                CompiledSubsection(
                    id=item.id,
                    title=item.title,
                    title_en=item.title_en,
                    articles=(*item.articles, compiled) if item.id == subsection_id else item.articles,
                    briefing_title=item.briefing_title,
                    briefing_summary=item.briefing_summary,
                    briefing_title_en=item.briefing_title_en,
                    briefing_summary_en=item.briefing_summary_en,
                )
                for item in subsection_results
            )
            subsection_results = list(subsection_results)
        subsection_results, section_compiled = _ensure_compiled_subsection_minimums(
            section,
            subsection_results,
            section_compiled,
            used_urls,
            ranked_by_section.get(section.id, []),
            articles,
        )
        sections.append(
            CompiledSection(
                id=section.id,
                title=section.title,
                title_en=section.title_en,
                icon=section.icon,
                articles=tuple(section_compiled),
                briefing_title=normalize_space(str(raw_briefing.get("title") or f"{section.title}总结")),
                briefing_summary=normalize_space(
                    str(raw_briefing.get("summary") or _fallback_section_briefing(section, []))
                ),
                briefing_title_en=normalize_space(str(raw_briefing.get("title_en") or f"{section.title_en} Briefing")),
                briefing_summary_en=normalize_space(
                    str(raw_briefing.get("summary_en") or _fallback_section_briefing_en(section, []))
                ),
                subsections=tuple(subsection_results),
            )
        )

    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_title_en=str(config.site.get("title_en") or "Information Daily"),
        site_subtitle=str(config.site.get("subtitle") or ""),
        site_subtitle_en=str(config.site.get("subtitle_en") or config.site.get("subtitle") or ""),
        edition_label=str(config.site.get("edition_label") or ""),
        edition_label_en=str(config.site.get("edition_label_en") or config.site.get("edition_label") or ""),
        profile_id=config.profile_id,
        briefing_title=normalize_space(str(briefing.get("title") or "AI 总结")),
        briefing_summary=normalize_space(str(briefing.get("summary") or _fallback_briefing([]))),
        briefing_title_en=normalize_space(str(briefing.get("title_en") or "AI Briefing")),
        briefing_summary_en=normalize_space(str(briefing.get("summary_en") or _fallback_briefing_en([]))),
        headline=headline,
        sections=tuple(sections),
        source_count=len([source for source in config.sources if source.enabled]),
        raw_count=len(articles),
        warnings=tuple(warnings),
        generated_at=_now(config),
    )


def _ranked_candidates_by_section(
    config: AppConfig,
    articles: list[Article],
) -> dict[str, list[tuple[Article, float]]]:
    by_section = {section.id: [] for section in config.sections}
    for article in articles:
        section, score = _best_section(config, article)
        by_section.setdefault(section.id, []).append((article, score))
    for items in by_section.values():
        items.sort(
            key=lambda row: (
                row[1],
                row[0].published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
            ),
            reverse=True,
        )
    return by_section


def _ranked_candidates_by_subsection(
    config: AppConfig,
    articles: list[Article],
) -> dict[str, dict[str, list[tuple[Article, float]]]]:
    by_subsection = {
        section.id: {subsection.id: [] for subsection in _section_subsections(section)}
        for section in config.sections
    }
    for article in articles:
        section, section_score = _best_section(config, article)
        subsection = _best_subsection(section, article)
        by_subsection[section.id][subsection.id].append((article, section_score))
    for section_items in by_subsection.values():
        for items in section_items.values():
            items.sort(
                key=lambda row: (
                    row[1],
                    row[0].published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
                ),
                reverse=True,
            )
    return by_subsection


def _llm_candidates_by_section(
    config: AppConfig,
    ranked_by_section: dict[str, list[tuple[Article, float]]],
) -> dict[str, list[tuple[Article, float]]]:
    total_limit = max(1, int(config.llm.max_input_items or 60))
    pools = {
        section.id: ranked_by_section.get(section.id, [])[
            : min(8, max(section.max_articles, 1))
        ]
        for section in config.sections
    }
    selected = {section.id: [] for section in config.sections}
    remaining = total_limit
    index = 0
    while remaining > 0:
        progressed = False
        for section in config.sections:
            pool = pools.get(section.id, [])
            if index >= len(pool):
                continue
            selected[section.id].append(pool[index])
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
        index += 1
    return selected


def _article_candidate_payload(article: Article, score: float, section: SectionConfig) -> dict:
    subsection = _best_subsection(section, article)
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source_name,
        "source_zh": _source_label_zh(article.source_name),
        "source_en": article.source_name,
        "summary": article.summary[:360],
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "suggested_subsection": subsection.id,
        "suggested_subsection_title": subsection.title,
        "score": round(score, 2),
    }


def _section_subsections(section: SectionConfig) -> tuple[SubsectionConfig, ...]:
    if section.subsections:
        return section.subsections
    return (
        SubsectionConfig(
            id=f"{section.id}-all",
            title=section.title,
            title_en=section.title_en,
            max_articles=section.max_articles,
            keywords=section.keywords,
        ),
    )


def _best_subsection(section: SectionConfig, article: Article) -> SubsectionConfig:
    subsections = _section_subsections(section)
    text = f"{article.title} {article.summary} {' '.join(article.keywords)}".lower()
    best = subsections[0]
    best_score = -1
    for subsection in subsections:
        score = sum(1 for keyword in subsection.keywords if keyword and keyword in text)
        if score > best_score:
            best = subsection
            best_score = score
    return best


def _compiled_subsections_from_pairs(
    section: SectionConfig,
    items: list[tuple[Article, float]],
) -> tuple[CompiledSubsection, ...]:
    grouped = {subsection.id: [] for subsection in _section_subsections(section)}
    for article, score in items:
        grouped[_best_subsection(section, article).id].append((article, score))
    return _compiled_subsections_from_grouped(section, grouped)


def _compiled_subsections_from_grouped(
    section: SectionConfig,
    grouped: dict[str, list[tuple[Article, float]]],
) -> tuple[CompiledSubsection, ...]:
    return tuple(
        CompiledSubsection(
            id=subsection.id,
            title=subsection.title,
            title_en=subsection.title_en,
            articles=tuple(_compiled_article(article, score) for article, score in grouped[subsection.id]),
            briefing_title=f"{subsection.title}总结",
            briefing_summary=_fallback_subsection_briefing(subsection, grouped[subsection.id]),
            briefing_title_en=f"{subsection.title_en} Briefing",
            briefing_summary_en=_fallback_subsection_briefing_en(subsection, grouped[subsection.id]),
        )
        for subsection in _section_subsections(section)
    )


def _balanced_subsection_pairs(
    section: SectionConfig,
    scored: list[tuple[Article, SectionConfig, float]],
    used_article_keys: set[str],
) -> dict[str, list[tuple[Article, float]]]:
    subsections = _section_subsections(section)
    grouped: dict[str, list[tuple[Article, float]]] = {subsection.id: [] for subsection in subsections}
    selected_keys: set[str] = set()
    section_rows = [
        (article, _score_article_for_section(section, article))
        for article, item_section, _ in scored
        if item_section.id == section.id
    ]
    if not section_rows:
        return grouped

    all_rows = [
        (article, _score_article_for_section(section, article))
        for article, _, _ in scored
    ]
    section_rows = _sorted_article_rows(section_rows)
    all_rows = _sorted_article_rows(all_rows)

    def can_add(article: Article, subsection: SubsectionConfig) -> bool:
        key = _article_key(article)
        return (
            key not in used_article_keys
            and key not in selected_keys
            and len(grouped[subsection.id]) < subsection.max_articles
        )

    def add_to(subsection: SubsectionConfig, row: tuple[Article, float]) -> bool:
        article, score = row
        if not can_add(article, subsection):
            return False
        grouped[subsection.id].append((article, score))
        selected_keys.add(_article_key(article))
        return True

    for subsection in subsections:
        target = _minimum_for_subsection(subsection)
        for rows, require_match in ((section_rows, True), (all_rows, True), (all_rows, False)):
            if len(grouped[subsection.id]) >= target:
                break
            for row in rows:
                if len(grouped[subsection.id]) >= target:
                    break
                if require_match and _best_subsection(section, row[0]).id != subsection.id:
                    continue
                add_to(subsection, row)

    section_target = max(section.max_articles, sum(_minimum_for_subsection(item) for item in subsections))
    for rows, allow_global in ((section_rows, False), (all_rows, True)):
        if _grouped_count(grouped) >= section_target:
            break
        for row in rows:
            if _grouped_count(grouped) >= section_target:
                break
            article = row[0]
            if not allow_global and _best_section_id_from_scored(scored, article) != section.id:
                continue
            subsection = _best_subsection(section, article)
            add_to(subsection, row)
    return grouped


def _minimum_for_subsection(subsection: SubsectionConfig) -> int:
    return min(MIN_SUBSECTION_ARTICLES, subsection.max_articles)


def _grouped_count(grouped: dict[str, list[tuple[Article, float]]]) -> int:
    return sum(len(items) for items in grouped.values())


def _sorted_article_rows(rows: list[tuple[Article, float]]) -> list[tuple[Article, float]]:
    return sorted(
        rows,
        key=lambda row: (
            row[1],
            row[0].published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
        ),
        reverse=True,
    )


def _best_section_id_from_scored(
    scored: list[tuple[Article, SectionConfig, float]],
    article: Article,
) -> str:
    key = _article_key(article)
    for scored_article, section, _ in scored:
        if _article_key(scored_article) == key:
            return section.id
    return ""


def _ensure_compiled_subsection_minimums(
    section: SectionConfig,
    subsection_results: list[CompiledSubsection],
    section_compiled: list[CompiledArticle],
    used_article_keys: set[str],
    section_rows: list[tuple[Article, float]],
    articles: list[Article],
) -> tuple[list[CompiledSubsection], list[CompiledArticle]]:
    if not section_rows and not section_compiled:
        return subsection_results, section_compiled

    by_id = {subsection.id: subsection for subsection in subsection_results}
    all_rows = _sorted_article_rows([
        (article, _score_article_for_section(section, article))
        for article in articles
    ])
    section_rows = _sorted_article_rows([
        (article, _score_article_for_section(section, article))
        for article, _ in section_rows
    ])

    def append_to(subsection: SubsectionConfig, row: tuple[Article, float]) -> bool:
        article, score = row
        key = _article_key(article)
        if key in used_article_keys:
            return False
        current = by_id[subsection.id]
        if len(current.articles) >= subsection.max_articles:
            return False
        compiled = _compiled_article(article, score)
        by_id[subsection.id] = CompiledSubsection(
            id=current.id,
            title=current.title,
            title_en=current.title_en,
            articles=(*current.articles, compiled),
            briefing_title=current.briefing_title,
            briefing_summary=current.briefing_summary,
            briefing_title_en=current.briefing_title_en,
            briefing_summary_en=current.briefing_summary_en,
        )
        section_compiled.append(compiled)
        used_article_keys.add(key)
        return True

    for subsection in _section_subsections(section):
        target = _minimum_for_subsection(subsection)
        for rows, require_match in ((section_rows, True), (all_rows, True), (all_rows, False)):
            current = by_id.get(subsection.id)
            if current is None or len(current.articles) >= target:
                break
            for row in rows:
                current = by_id[subsection.id]
                if len(current.articles) >= target:
                    break
                if require_match and _best_subsection(section, row[0]).id != subsection.id:
                    continue
                append_to(subsection, row)

    ordered = []
    for subsection in _section_subsections(section):
        item = by_id.get(subsection.id)
        if item is not None:
            ordered.append(item)
    return ordered, section_compiled


def _compiled_from_mapping(value: object, allowed_by_url: dict[str, Article]) -> CompiledArticle | None:
    item, _ = _compiled_from_mapping_checked(value, allowed_by_url)
    return item


def _compiled_from_mapping_checked(
    value: object,
    allowed_by_url: dict[str, Article],
) -> tuple[CompiledArticle | None, str]:
    if not isinstance(value, dict):
        return None, "item was not an object"
    url = str(value.get("url") or "")
    if not url:
        return None, "missing URL"
    source_article = allowed_by_url.get(url)
    if source_article is None:
        return None, f"unknown URL: {url}"
    return CompiledArticle(
        title=normalize_space(str(value.get("title") or _fallback_article_title_zh(source_article))),
        url=url,
        source=_source_label_zh(source_article.source_name),
        summary=normalize_space(str(value.get("summary") or _fallback_article_summary_zh(source_article))),
        source_en=source_article.source_name,
        title_en=normalize_space(str(value.get("title_en") or source_article.title)),
        summary_en=normalize_space(str(value.get("summary_en") or source_article.summary or source_article.title)),
        original_title=source_article.title,
        published_at=source_article.published_at.isoformat() if source_article.published_at else None,
        reason=normalize_space(str(value.get("reason") or "")),
        reason_en=normalize_space(str(value.get("reason_en") or value.get("reason") or "")),
        score=_float_value(value.get("score")),
    ), ""


def _best_section(config: AppConfig, article: Article) -> tuple[SectionConfig, float]:
    best = config.sections[0]
    best_score = -1.0
    for section in config.sections:
        score = _score_article_for_section(section, article)
        if score > best_score:
            best = section
            best_score = score
    return best, best_score


def _score_article_for_section(section: SectionConfig, article: Article) -> float:
    text = f"{article.title} {article.summary} {' '.join(article.keywords)}".lower()
    score = article.weight
    if article.default_section == section.id:
        score += 2.0
    score += sum(1.0 for keyword in section.keywords if keyword and keyword in text)
    return score


def _article_key(article: Article) -> str:
    return article.url or article.id or normalize_space(article.title).lower()


def _article_key_from_compiled(article: CompiledArticle | None) -> str:
    if article is None:
        return ""
    return article.url or normalize_space(article.title).lower()


def _compiled_article(article: Article, score: float) -> CompiledArticle:
    summary = article.summary or article.title
    if len(summary) > 220:
        summary = summary[:217].rstrip() + "..."
    return CompiledArticle(
        title=_fallback_article_title_zh(article),
        url=article.url,
        source=_source_label_zh(article.source_name),
        summary=_fallback_article_summary_zh(article),
        source_en=article.source_name,
        title_en=article.title,
        summary_en=summary,
        original_title=article.title,
        published_at=article.published_at.isoformat() if article.published_at else None,
        score=round(score, 2),
    )


def _float_value(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _fallback_article_title_zh(article: Article) -> str:
    topic = _fallback_topic_label(article)
    return f"{topic}：{_source_label_zh(article.source_name)}关注的关键线索"


def _fallback_article_summary_zh(article: Article) -> str:
    topic = _fallback_topic_label(article)
    source = _source_label_zh(article.source_name)
    if topic == "财经市场":
        return (
            f"这条财经线索来自{source}，主要涉及市场走势、资产定价或宏观预期变化，"
            "可作为观察风险偏好、利率环境和投资节奏的参考。"
        )
    if topic == "中国观察":
        return (
            f"这条中国观察线索来自{source}，关注国内政策、产业变化或社会治理动态，"
            "适合用来判断本土环境和公共议题的变化。"
        )
    if topic == "国际事务":
        return (
            f"这条国际线索来自{source}，重点关注地缘安全、政府政策或全球社会动态，"
            "有助于判断外部环境和区域风险的变化。"
        )
    if topic == "开发工具":
        return (
            f"这条开发线索来自{source}，围绕工程实践、开源工具或开发者生态展开，"
            "适合关注工具链效率和技术采用趋势。"
        )
    if topic == "研究进展":
        return (
            f"这条研究线索来自{source}，关注论文、基准或实验方法的新进展，"
            "可用于跟踪相关领域的技术演化。"
        )
    if topic == "科技科学":
        return (
            f"这条科技科学线索来自{source}，关注芯片硬件、航天能源、生命科学或基础研究，"
            "有助于从 AI 之外观察技术和科学进展。"
        )
    if topic == "创业产品":
        return (
            f"这条创业产品线索来自{source}，聚焦新产品、融资并购或增长策略，"
            "适合观察早期市场机会和产品化趋势。"
        )
    if topic == "观点评论":
        return (
            f"这条观点线索来自{source}，更偏分析、讨论或长期判断，"
            "适合放在事实新闻之后补充理解框架。"
        )
    return (
        f"这条科技线索来自{source}，关注模型能力、产品落地或产业应用变化，"
        "适合快速把握今日技术方向。"
    )


def _fallback_topic_label(article: Article) -> str:
    if article.default_section == "ai-tech":
        return "AI 与大模型"
    if article.default_section == "developer":
        return "开发工具"
    if article.default_section == "science-tech":
        return "科技科学"
    if article.default_section == "startup-product":
        return "创业产品"
    if article.default_section == "world":
        return "国际事务"
    if article.default_section == "china":
        return "中国观察"
    if article.default_section == "finance":
        return "财经市场"
    if article.default_section == "opinion":
        return "观点评论"
    text = f"{article.default_section} {article.title} {article.summary} {' '.join(article.keywords)}".lower()
    if any(keyword in text for keyword in ("llm", "openai", "anthropic", "model", "agent", "ai")):
        return "AI 与大模型"
    if any(keyword in text for keyword in ("developer", "github", "programming", "python", "javascript", "tool")):
        return "开发工具"
    if any(keyword in text for keyword in ("science", "space", "nasa", "chip", "hardware", "robotics", "energy")):
        return "科技科学"
    if any(keyword in text for keyword in ("startup", "product", "funding", "venture", "founder", "growth")):
        return "创业产品"
    if any(keyword in text for keyword in ("arxiv", "paper", "research", "benchmark")):
        return "研究进展"
    if any(keyword in text for keyword in ("china", "中国", "国内", "时政", "政策", "社会", "法治")):
        return "中国观察"
    if any(keyword in text for keyword in ("war", "geopolitics", "diplomacy", "security", "election", "government")):
        return "国际事务"
    if any(keyword in text for keyword in ("market", "stock", "bond", "fed", "inflation", "economy", "finance")):
        return "财经市场"
    if any(keyword in text for keyword in ("opinion", "analysis", "essay", "editorial", "discussion", "hacker news")):
        return "观点评论"
    return "新闻"


def _source_label_zh(source_name: str) -> str:
    labels = {
        "Al Jazeera": "半岛电视台",
        "BBC World": "BBC 国际",
        "Bloomberg Economics": "彭博经济",
        "Bloomberg Markets": "彭博市场",
        "Federal Reserve": "美联储",
        "GitHub Blog": "GitHub 博客",
        "Google AI Blog": "Google AI 博客",
        "Hacker News": "Hacker News",
        "Hugging Face Blog": "Hugging Face 博客",
        "Investing.com": "英为财情",
        "MarketWatch": "市场观察",
        "Meta Engineering": "Meta 工程",
        "NASA Technology": "NASA 技术",
        "NPR World": "NPR 国际",
        "OpenAI News": "OpenAI 新闻",
        "Product Hunt": "Product Hunt",
        "ScienceDaily Top Science": "ScienceDaily 科学",
        "ScienceDaily Top Technology": "ScienceDaily 技术",
        "SEC Press Releases": "美国证交会新闻",
        "Simon Willison": "Simon Willison",
        "TechCrunch AI": "TechCrunch AI",
        "TechCrunch Startups": "TechCrunch 创业",
        "The Guardian World": "卫报国际",
        "Vercel Blog": "Vercel 博客",
        "WSJ Markets": "华尔街日报市场",
        "arXiv cs.AI": "arXiv 人工智能",
        "arXiv cs.CL": "arXiv 计算语言学",
        "arXiv cs.LG": "arXiv 机器学习",
        "中国新闻网时政": "中国新闻网时政",
        "中国新闻网社会": "中国新闻网社会",
        "新华网时政": "新华网时政",
        "新华网科技": "新华网科技",
    }
    return labels.get(source_name, source_name)


def _fallback_briefing(scored: list[tuple[Article, SectionConfig, float]]) -> str:
    if not scored:
        return "今日暂未获取到足够新闻。请检查数据源、关键词或 LLM 配置后重新生成。"
    top_sources = []
    for article, _, _ in scored[:5]:
        source = _source_label_zh(article.source_name)
        if source not in top_sources:
            top_sources.append(source)
    top_sections = []
    for _, section, _ in scored:
        if section.title not in top_sections:
            top_sections.append(section.title)
        if len(top_sections) >= 4:
            break
    return (
        f"今日本报汇总 {len({section.id for _, section, _ in scored})} 个版面的信息，"
        f"重点覆盖{'、'.join(top_sections)}等方向，帮助快速把握技术、商业、公共议题和市场变化。"
        f" 本期共整理 {len(scored)} 条候选线索，主要来源包括：{'、'.join(top_sources)}。"
    )


def _fallback_briefing_en(scored: list[tuple[Article, SectionConfig, float]]) -> str:
    if not scored:
        return "Not enough news items were collected today. Check sources, keywords, or LLM settings and generate again."
    top_sources = []
    for article, _, _ in scored[:5]:
        if article.source_name not in top_sources:
            top_sources.append(article.source_name)
    return (
        "This is a local rule-based preview without formal LLM editing. "
        "In production, the LLM will summarize the main themes, translate where needed, "
        "and prepare both Chinese and English editions. "
        f"This preview read {len(scored)} candidate items from sources including: {', '.join(top_sources)}."
    )


def _fallback_section_briefing(
    section: SectionConfig,
    items: list[tuple[Article, float]],
) -> str:
    if not items:
        return f"{section.title}暂无足够内容生成总结。请检查该类别的数据源、关键词或 LLM 输出。"
    sources = _source_list_zh(article for article, _ in items[:5])
    return (
        f"本版聚焦 {section.title}，汇集 {len(items)} 条相关线索，主要来源包括：{sources}。"
        "阅读时可优先关注事件之间的共振关系，以及它们对后续趋势的影响。"
    )


def _fallback_section_briefing_en(
    section: SectionConfig,
    items: list[tuple[Article, float]],
) -> str:
    if not items:
        return f"There is not enough content to summarize {section.title_en}. Check sources, keywords, or LLM output."
    titles = "; ".join(article.title for article, _ in items[:4])
    return (
        f"This edition focuses on {section.title_en}. The local rule-based preview has not been formally edited by the LLM. "
        f"Key items include: {titles}."
    )


def _fallback_subsection_briefing(
    subsection: SubsectionConfig,
    items: list[tuple[Article, float]],
) -> str:
    if not items:
        return f"{subsection.title}暂无足够内容生成总结。"
    sources = _source_list_zh(article for article, _ in items[:4])
    return f"{subsection.title}聚焦本组最相关的新闻线索，共 {len(items)} 条，主要来源包括：{sources}。"


def _source_list_zh(articles) -> str:
    sources = []
    for article in articles:
        if article.source_name not in sources:
            sources.append(_source_label_zh(article.source_name))
    return "、".join(sources) if sources else "暂无"


def _fallback_subsection_briefing_en(
    subsection: SubsectionConfig,
    items: list[tuple[Article, float]],
) -> str:
    if not items:
        return f"There is not enough content to summarize {subsection.title_en}."
    titles = "; ".join(article.title for article, _ in items[:3])
    return f"{subsection.title_en} focuses on the most relevant items in this group. Key items include: {titles}."


def _now(config: AppConfig) -> datetime:
    timezone_name = str(config.site.get("timezone") or "UTC")
    return datetime.now(ZoneInfo(timezone_name))


def issue_to_dict(issue: Issue) -> dict:
    return asdict(issue)
