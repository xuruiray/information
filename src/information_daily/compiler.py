from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, replace
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
            warnings.append(
                "LLM credentials are missing; generated with rule-based fallback."
            )
        except CompileError:
            if not allow_fallback:
                raise
            warnings.append(
                "LLM compilation failed; generated with rule-based fallback."
            )

    issue = _compile_with_rules(config, prepared, issue_date)
    return Issue(
        **{
            field: getattr(issue, field)
            for field in issue.__dataclass_fields__
            if field != "warnings"
        },
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
        issue_end = datetime.combine(
            issue_date + timedelta(days=1), time.min, tzinfo=timezone
        )
        cutoff = issue_end - timedelta(hours=float(lookback_hours))

    seen: set[str] = set()
    deduped: list[Article] = []
    for article in articles:
        if (
            cutoff
            and article.published_at
            and article.published_at.astimezone(timezone) < cutoff
        ):
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


def _compile_with_rules(
    config: AppConfig, articles: list[Article], issue_date: date
) -> Issue:
    scored: list[tuple[Article, SectionConfig, float]] = []
    for article in articles:
        section, score = _best_section(config, article)
        if score >= float(config.selection.get("min_score") or 0):
            scored.append((article, section, score))
    scored.sort(
        key=lambda row: (
            row[2],
            row[0].published_at or datetime.min.replace(tzinfo=ZoneInfo("UTC")),
        ),
        reverse=True,
    )

    sections: list[CompiledSection] = []
    used_urls = set()
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
                articles=tuple(
                    _compiled_article(article, score) for article, score in items
                ),
                briefing_title=f"{section.title}总结",
                briefing_summary=_article_brief(
                    [_compiled_article(a, score) for a, score in items]
                ),
                briefing_title_en=f"{section.title_en} Briefing",
                briefing_summary_en=_article_brief(
                    [_compiled_article(a, score) for a, score in items], lang="en"
                ),
                subsections=subsections,
            )
        )

    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_title_en=str(config.site.get("title_en") or "Information Daily"),
        site_subtitle=str(config.site.get("subtitle") or ""),
        site_subtitle_en=str(
            config.site.get("subtitle_en") or config.site.get("subtitle") or ""
        ),
        edition_label=str(config.site.get("edition_label") or ""),
        edition_label_en=str(
            config.site.get("edition_label_en")
            or config.site.get("edition_label")
            or ""
        ),
        profile_id=config.profile_id,
        briefing_title="AI 总结",
        briefing_summary=_article_brief(
            [a for section in sections for a in section.articles]
        ),
        briefing_title_en="AI Briefing",
        briefing_summary_en=_article_brief(
            [a for section in sections for a in section.articles], lang="en"
        ),
        headline=next((a for section in sections for a in section.articles), None),
        sections=tuple(sections),
        source_count=len([source for source in config.sources if source.enabled]),
        raw_count=len(articles),
        warnings=(),
        generated_at=_now(config),
    )


def _compile_with_llm(
    config: AppConfig, articles: list[Article], issue_date: date
) -> Issue:
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise ConfigError(f"Missing required LLM secret: {config.llm.api_key_env}")

    ranked = _ranked_candidates_by_section(config, articles)
    section_data = []
    for section in config.sections:
        # Selection has no minimum; the editor may reject every candidate.
        section_data.append(
            _compile_llm_section(config, section, ranked, issue_date, api_key)
        )
    issue = _issue_from_llm_data(
        config, {"sections": section_data}, articles, issue_date
    )
    sections = tuple(
        _compile_llm_briefings(config, section, issue_date, api_key)
        for section in issue.sections
    )
    issue = replace(issue, sections=sections)
    if not any(section.articles for section in sections):
        raise CompileError(
            "No publishable articles remained after editorial validation"
        )
    overview = _compile_llm_overview(config, sections, issue_date, api_key)
    return _apply_final_overview(issue, overview)


def _apply_final_overview(issue: Issue, overview: dict) -> Issue:
    selected_by_url = {
        item.url: item for section in issue.sections for item in section.articles
    }
    headline_url = str(overview.get("headline_url") or "")
    if headline_url not in selected_by_url:
        raise CompileError("Overview headline must reference a final published article")
    briefing = _validated_briefing(
        overview.get("briefing"), set(selected_by_url), "overview"
    )
    return replace(
        issue,
        headline=selected_by_url[headline_url],
        briefing_title=briefing["title"],
        briefing_summary=briefing["summary"],
        briefing_title_en=briefing["title_en"],
        briefing_summary_en=briefing["summary_en"],
    )


def _compile_llm_section(config, section, ranked, issue_date, api_key) -> dict:
    limit = min(max(1, config.llm.max_input_items), max(1, section.max_articles * 2))
    candidates = [
        _article_candidate_payload(article, score, section)
        for article, score in ranked.get(section.id, [])[:limit]
    ]
    if not candidates:
        return {"id": section.id, "subsections": []}
    request = {
        "task": f"选择并编写 {section.title} 的文章；不要生成总结。",
        "date": issue_date.isoformat(),
        "style": config.llm.summary_style,
        "section": {
            "id": section.id,
            "title": section.title,
            "max_articles": section.max_articles,
            "keywords": section.keywords,
            "subsections": [
                {
                    "id": sub.id,
                    "title": sub.title,
                    "keywords": sub.keywords,
                    "max_articles": sub.max_articles,
                }
                for sub in _section_subsections(section)
            ],
        },
        "other_sections": [
            {"id": item.id, "title": item.title}
            for item in config.sections
            if item.id != section.id
        ],
        "requirements": [
            "只选择主题直接符合本大版和所在小版的文章。根据标题和摘要的实际事件判断；来源名称和 suggested_subsection 只是提示，不能作为归属依据。",
            "没有最低篇数要求；max_articles 只是上限。没有相关且有信息量的文章时返回空列表，不要从其他主题补齐。",
            "一条新闻只放入一个小版。文章 URL 必须来自 candidates，不能改写。",
            "科技公司涉及的战争、外交、选举新闻不能仅因公司名归入开发或 AI；论文不因出现 model 就自动属于前沿模型。",
            "标题和摘要必须描述具体事件、方法、结果或变化；不写某来源关注的关键线索、适合关注技术方向等通用文案。",
            "中文 title 和 summary 必须完整；无法可靠翻译或材料不足则跳过。保留不确定性，不增加材料中没有的事实。",
            "每篇提供 topic_evidence：从原始 title 或 summary 中逐字摘取含该小版 keywords 的主题短语，不使用来源标签；没有直接主题证据时跳过。",
        ],
        "schema": {
            "id": section.id,
            "subsections": [
                {
                    "id": "subsection id",
                    "articles": [
                        {
                            "url": "candidate URL",
                            "title": "具体中文标题",
                            "summary": "50-110字具体中文摘要",
                            "title_en": "English title",
                            "summary_en": "English summary",
                            "topic_evidence": "verbatim topic evidence from candidate title or summary",
                            "reason": "具体入选理由",
                            "reason_en": "selection rationale",
                            "score": 0,
                        }
                    ],
                }
            ],
        },
        "candidates": candidates,
    }
    data = _request_editor(config, request, api_key)
    return _coerce_llm_section_data(section, data)


def _compiled_payload(article: CompiledArticle) -> dict:
    return {
        "url": article.url,
        "title": article.title,
        "summary": article.summary,
        "title_en": article.title_en,
        "summary_en": article.summary_en,
        "source": article.source,
    }


def _briefing_schema() -> dict:
    return {
        "title": "中文总结标题",
        "summary": "基于最终文章的简洁中文总结",
        "title_en": "English briefing title",
        "summary_en": "Briefing based on final articles",
        "covered_urls": ["every URL in the input article set, unchanged"],
    }


def _request_editor(config, request, api_key) -> dict:
    payload = _llm_payload(
        config,
        [
            {"role": "system", "content": _llm_system_prompt()},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
    )
    return _extract_json(_post_chat_completion(config, payload, api_key))


def _validated_briefing(value, urls: set[str], scope: str) -> dict:
    if not isinstance(value, dict):
        raise CompileError(f"Missing final briefing: {scope}")
    covered = value.get("covered_urls")
    if (
        not isinstance(covered, list)
        or any(not isinstance(url, str) for url in covered)
        or set(covered) != urls
    ):
        raise CompileError(
            f"Final briefing article set differs from published articles: {scope}"
        )
    fields = {
        key: normalize_space(str(value.get(key) or ""))
        for key in ("title", "summary", "title_en", "summary_en")
    }
    if any(not text for text in fields.values()) or not _has_chinese(fields["summary"]):
        raise CompileError(f"Incomplete bilingual final briefing: {scope}")
    if _generic_copy(fields["summary"]) or any(
        phrase in fields["summary"]
        for phrase in (
            "暂无文章",
            "暂无入选",
            "无文章入选",
            "没有文章入选",
            "暂无新闻",
            "暂无符合",
            "无候选文章",
            "没有入选",
            "暂无足够",
        )
    ):
        raise CompileError(f"Final briefing contradicts nonempty articles: {scope}")
    return fields


def _compile_llm_briefings(
    config, section: CompiledSection, issue_date, api_key
) -> CompiledSection:
    if not section.articles:
        return section
    nonempty = [sub for sub in section.subsections if sub.articles]
    request = {
        "task": f"根据最终文章生成 {section.title} 及小版总结。",
        "date": issue_date.isoformat(),
        "section": {"id": section.id, "title": section.title},
        "final_subsections": [
            {
                "id": sub.id,
                "title": sub.title,
                "articles": [_compiled_payload(a) for a in sub.articles],
            }
            for sub in nonempty
        ],
        "requirements": [
            "文章筛选、去重和栏目归属已经完成；只总结输入中的最终文章，不新增、删减、重新选题或移动文章。",
            "总结描述具体事件和变化，不泛谈行业方向。保留报道、预测、论文提出等事实限定。",
            "每个 covered_urls 必须列出对应总结的全部输入文章 URL；大版列出全部小版文章 URL。",
            "输入只有非空小版，不要写暂无文章或无重大更新，不得提及未入选报道。",
        ],
        "schema": {
            "briefing": _briefing_schema(),
            "subsections": [
                {"id": "input subsection id", "briefing": _briefing_schema()}
            ],
        },
    }
    for attempt in range(2):
        data = _request_editor(config, request, api_key)
        try:
            return _apply_final_briefings(section, data)
        except CompileError as exc:
            if attempt:
                raise
            request["repair"] = str(exc)
    raise CompileError("Could not generate final briefings")


def _apply_final_briefings(section: CompiledSection, data: dict) -> CompiledSection:
    nonempty = [sub for sub in section.subsections if sub.articles]
    briefing = _validated_briefing(
        data.get("briefing"), {a.url for a in section.articles}, section.id
    )
    raw_subs = data.get("subsections")
    if not isinstance(raw_subs, list) or any(
        not isinstance(sub, dict) for sub in raw_subs
    ):
        raise CompileError(f"Missing final subsection briefings: {section.id}")
    by_id = {sub.get("id"): sub for sub in raw_subs if isinstance(sub.get("id"), str)}
    if len(by_id) != len(raw_subs) or set(by_id) != {sub.id for sub in nonempty}:
        raise CompileError(f"Final briefing subsection set differs: {section.id}")
    subs = []
    for sub in section.subsections:
        if not sub.articles:
            subs.append(sub)
            continue
        fields = _validated_briefing(
            by_id[sub.id].get("briefing"),
            {a.url for a in sub.articles},
            f"{section.id}/{sub.id}",
        )
        subs.append(_with_briefing(sub, fields))
    return _with_briefing(replace(section, subsections=tuple(subs)), briefing)


def _with_briefing(item, fields):
    return replace(
        item,
        briefing_title=fields["title"],
        briefing_summary=fields["summary"],
        briefing_title_en=fields["title_en"],
        briefing_summary_en=fields["summary_en"],
    )


def _compile_llm_overview(config, sections, issue_date, api_key) -> dict:
    request = {
        "task": "根据最终发布文章生成整份日报的总览，并选择头条 URL。",
        "date": issue_date.isoformat(),
        "final_sections": [
            {
                "id": section.id,
                "title": section.title,
                "articles": [_compiled_payload(a) for a in section.articles],
            }
            for section in sections
            if section.articles
        ],
        "requirements": [
            "只总结最终发布文章，突出具体变化和阅读重点；不得提及未入选候选，也不重新改写文章。",
            "headline_url 必须来自 final_sections 中的文章，covered_urls 列出全部最终文章 URL。",
            "中文总结约160-260字，英文简洁；区分已发生事实、研究主张和预测，保留不确定性。",
        ],
        "schema": {
            "briefing": _briefing_schema(),
            "headline_url": "URL of one final article",
        },
    }
    urls = {a.url for section in sections for a in section.articles}
    for attempt in range(2):
        data = _request_editor(config, request, api_key)
        try:
            _validated_briefing(data.get("briefing"), urls, "overview")
            if data.get("headline_url") not in urls:
                raise CompileError(
                    "Overview headline must reference a final published article"
                )
            return data
        except CompileError as exc:
            if attempt:
                raise
            request["repair"] = str(exc)
    raise CompileError("Could not generate final overview")


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
    raise CompileError(f"Selection response does not match section: {section.id}")


def _post_chat_completion(config: AppConfig, payload: dict, api_key: str) -> str:
    base_url = os.environ.get(
        config.llm.base_url_env, config.llm.default_base_url
    ).rstrip("/")
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
        raise CompileError(
            "LLM response did not include choices[0].message.content"
        ) from exc


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


def _issue_from_llm_data(config, data, articles, issue_date) -> Issue:
    ranked = _ranked_candidates_by_section(config, articles)
    raw_sections = {
        item["id"]: item
        for item in data.get("sections", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    sections = []
    warnings = []
    used_urls = set()
    for section in config.sections:
        limit = min(
            max(1, config.llm.max_input_items), max(1, section.max_articles * 2)
        )
        allowed = {a.url: a for a, _ in ranked[section.id][:limit]}
        raw_subsections = raw_sections.get(section.id, {}).get("subsections", [])
        if not isinstance(raw_subsections, list):
            raw_subsections = []
        by_id = {}
        known_ids = {sub.id for sub in _section_subsections(section)}
        for raw_sub in raw_subsections:
            if (
                not isinstance(raw_sub, dict)
                or not isinstance(raw_sub.get("id"), str)
                or raw_sub["id"] not in known_ids
            ):
                warnings.append(f"Unknown subsection ignored in {section.id}")
                continue
            by_id.setdefault(raw_sub["id"], []).extend(
                raw_sub.get("articles", [])
                if isinstance(raw_sub.get("articles"), list)
                else []
            )
        compiled_subsections = []
        section_articles = []
        for sub in _section_subsections(section):
            compiled = []
            for raw_item in by_id.get(sub.id, []):
                if (
                    len(compiled) >= sub.max_articles
                    or len(section_articles) + len(compiled) >= section.max_articles
                ):
                    break
                item, warning = _compiled_from_mapping_checked(raw_item, allowed)
                if item is not None:
                    original = allowed[item.url]
                    evidence = normalize_space(
                        str(raw_item.get("topic_evidence") or "")
                    )
                    if not evidence or not any(
                        evidence.casefold() in text.casefold()
                        for text in (original.title, original.summary)
                    ):
                        warning = "missing verbatim topic evidence"
                    elif not _topic_matches(sub.keywords, evidence):
                        warning = "topic evidence does not match subsection"
                    elif item.url in used_urls:
                        warning = "duplicate URL"
                    if not warning:
                        compiled.append(item)
                        used_urls.add(item.url)
                if warning:
                    warnings.append(
                        f"Editorial item ignored in {section.id}/{sub.id}: {warning}"
                    )
            compiled_subsections.append(
                CompiledSubsection(
                    id=sub.id,
                    title=sub.title,
                    title_en=sub.title_en,
                    articles=tuple(compiled),
                    briefing_title=f"{sub.title}总结",
                    briefing_title_en=f"{sub.title_en} Briefing",
                    briefing_summary=_article_brief(compiled),
                    briefing_summary_en=_article_brief(compiled, lang="en"),
                )
            )
            section_articles.extend(compiled)
        sections.append(
            CompiledSection(
                id=section.id,
                title=section.title,
                title_en=section.title_en,
                icon=section.icon,
                articles=tuple(section_articles),
                subsections=tuple(compiled_subsections),
                briefing_title=f"{section.title}总结",
                briefing_title_en=f"{section.title_en} Briefing",
                briefing_summary=_article_brief(section_articles),
                briefing_summary_en=_article_brief(section_articles, lang="en"),
            )
        )
    return Issue(
        date=issue_date,
        site_title=str(config.site["title"]),
        site_title_en=str(config.site.get("title_en") or "Information Daily"),
        site_subtitle=str(config.site.get("subtitle") or ""),
        site_subtitle_en=str(
            config.site.get("subtitle_en") or config.site.get("subtitle") or ""
        ),
        edition_label=str(config.site.get("edition_label") or ""),
        edition_label_en=str(config.site.get("edition_label_en") or ""),
        profile_id=config.profile_id,
        briefing_title="今日总结",
        briefing_summary="",
        briefing_title_en="Daily Briefing",
        briefing_summary_en="",
        headline=None,
        sections=tuple(sections),
        source_count=sum(source.enabled for source in config.sources),
        raw_count=len(articles),
        warnings=tuple(warnings),
        generated_at=_now(config),
    )


def _article_brief(articles, lang="zh") -> str:
    if not articles:
        return "今日暂无入选文章。" if lang == "zh" else "No articles selected today."
    titles = [a.title if lang == "zh" else a.title_en for a in articles]
    return ("；" if lang == "zh" else "; ").join(titles) + (
        "。" if lang == "zh" else "."
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


def _article_candidate_payload(
    article: Article, score: float, section: SectionConfig
) -> dict:
    subsection = _best_subsection(section, article)
    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source_name,
        "source_zh": _source_label_zh(article.source_name),
        "source_en": article.source_name,
        "summary": article.summary[:360],
        "published_at": article.published_at.isoformat()
        if article.published_at
        else None,
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
    text = f"{article.title} {article.summary}".lower()
    best = subsections[0]
    best_score = -1
    for subsection in subsections:
        score = sum(
            1 for keyword in subsection.keywords if _keyword_matches(keyword, text)
        )
        if score > best_score:
            best = subsection
            best_score = score
    return best


def _compiled_subsections_from_grouped(
    section: SectionConfig,
    grouped: dict[str, list[tuple[Article, float]]],
) -> tuple[CompiledSubsection, ...]:
    return tuple(
        CompiledSubsection(
            id=subsection.id,
            title=subsection.title,
            title_en=subsection.title_en,
            articles=tuple(
                _compiled_article(article, score)
                for article, score in grouped[subsection.id]
            ),
            briefing_title=f"{subsection.title}总结",
            briefing_summary=_article_brief(
                [_compiled_article(a, score) for a, score in grouped[subsection.id]]
            ),
            briefing_title_en=f"{subsection.title_en} Briefing",
            briefing_summary_en=_article_brief(
                [_compiled_article(a, score) for a, score in grouped[subsection.id]],
                lang="en",
            ),
        )
        for subsection in _section_subsections(section)
    )


def _balanced_subsection_pairs(section, scored, used_article_keys):
    grouped = {sub.id: [] for sub in _section_subsections(section)}
    selected = set(used_article_keys)
    for article, item_section, score in scored:
        if item_section.id != section.id or _article_key(article) in selected:
            continue
        sub = _best_subsection(section, article)
        if not _topic_matches(sub.keywords, f"{article.title} {article.summary}"):
            continue
        if sum(len(items) for items in grouped.values()) >= section.max_articles:
            break
        if len(grouped[sub.id]) >= sub.max_articles:
            continue
        grouped[sub.id].append((article, score))
        selected.add(_article_key(article))
    return grouped


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
    title = normalize_space(str(value.get("title") or ""))
    summary = normalize_space(str(value.get("summary") or ""))
    if not title or not summary or not _has_chinese(title) or not _has_chinese(summary):
        return None, "missing concrete Chinese title or summary"
    if _generic_copy(title) or _generic_copy(summary):
        return None, "generic template copy"
    return CompiledArticle(
        title=title,
        url=url,
        source=_source_label_zh(source_article.source_name),
        summary=summary,
        source_en=source_article.source_name,
        title_en=normalize_space(str(value.get("title_en") or source_article.title)),
        summary_en=normalize_space(
            str(
                value.get("summary_en")
                or source_article.summary
                or source_article.title
            )
        ),
        original_title=source_article.title,
        published_at=source_article.published_at.isoformat()
        if source_article.published_at
        else None,
        reason=normalize_space(str(value.get("reason") or "")),
        reason_en=normalize_space(
            str(value.get("reason_en") or value.get("reason") or "")
        ),
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
    text = f"{article.title} {article.summary}".lower()
    score = article.weight
    if article.default_section == section.id:
        score += 2.0
    score += sum(1.0 for keyword in section.keywords if _keyword_matches(keyword, text))
    return score


def _article_key(article: Article) -> str:
    return article.url or article.id or normalize_space(article.title).lower()


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


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _generic_copy(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "关注的关键线索",
            "这条科技线索来自",
            "这条财经线索来自",
            "这条国际线索来自",
            "这条开发线索来自",
            "这条研究线索来自",
            "这条科技科学线索来自",
            "这条创业产品线索来自",
            "这条观点线索来自",
            "这条中国观察线索来自",
            "适合快速把握今日技术方向",
        )
    )


def _keyword_matches(keyword: str, text: str) -> bool:
    if not keyword or keyword.casefold() in {"arxiv", "blog", "hn", "hacker news"}:
        return False
    if _has_chinese(keyword):
        return keyword.casefold() in text.casefold()
    return bool(
        re.search(
            r"(?<![a-z0-9_])"
            + re.escape(keyword.casefold())
            + (
                r"(?:s)?"
                if keyword.isascii() and keyword.isalpha() and len(keyword) >= 4
                else ""
            )
            + r"(?![a-z0-9_])",
            text.casefold(),
        )
    )


def _topic_matches(keywords, text) -> bool:
    return not keywords or any(_keyword_matches(keyword, text) for keyword in keywords)


def _fallback_article_title_zh(article: Article) -> str:
    # A rule-only preview preserves the original text instead of fabricating a translation.
    return article.title


def _fallback_article_summary_zh(article: Article) -> str:
    return article.summary or article.title


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


def _now(config: AppConfig) -> datetime:
    timezone_name = str(config.site.get("timezone") or "UTC")
    return datetime.now(ZoneInfo(timezone_name))


def issue_to_dict(issue: Issue) -> dict:
    return asdict(issue)
