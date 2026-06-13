import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from shutil import copytree

import pytest

from information_daily.compiler import compile_issue
from information_daily.config import ConfigError, load_config
from information_daily.models import Article
from information_daily.renderer import render_issue
from information_daily.sources import FetchResult, SourceFetchStatus


def test_missing_llm_key_fails_without_fallback(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.delenv(config.llm.api_key_env, raising=False)

    with pytest.raises(ConfigError):
        compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=False)


def test_fallback_compiles_dynamic_sections(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.delenv(config.llm.api_key_env, raising=False)

    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=True)

    assert issue.headline is not None
    assert [section.id for section in issue.sections] == [section.id for section in config.sections]
    assert any(section.articles for section in issue.sections if section.id == "ai-tech")
    assert all(section.briefing_summary for section in issue.sections)
    assert all(len(section.subsections) == 3 for section in issue.sections)
    assert all(subsection.briefing_summary for section in issue.sections for subsection in section.subsections)
    ai_articles = next(section.articles for section in issue.sections if section.id == "ai-tech")
    assert ai_articles[0].title.startswith("AI 科技：")
    assert "适合" in ai_articles[0].summary or "关注" in ai_articles[0].summary
    assert "LLM" not in ai_articles[0].summary
    assert "规则预览" not in ai_articles[0].summary
    assert ai_articles[0].title_en in {
        "New LLM benchmark compares agent coding performance",
        "Useful open source CLI for developers",
    }
    assert issue.warnings


def test_lookback_hours_filters_old_articles(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.delenv(config.llm.api_key_env, raising=False)
    old_article = Article(
        id="old",
        title="Old AI story",
        url="https://example.com/old",
        source_id="old-source",
        source_name="Old Source",
        default_section="ai-tech",
        summary="Too old for the configured lookback window.",
        published_at=datetime(2026, 5, 20, 1, 0, tzinfo=timezone.utc),
    )

    issue = compile_issue(
        config,
        [*_articles(), old_article],
        date(2026, 5, 29),
        allow_fallback=True,
    )

    urls = {article.url for section in issue.sections for article in section.articles}
    assert old_article.url not in urls
    assert issue.raw_count == len(_articles())


def test_llm_compiles_overview_and_sections_separately(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    calls = []

    def fake_post_chat_completion(config, payload, api_key):
        del config, api_key
        request = json.loads(payload["messages"][1]["content"])
        calls.append(request["task"])
        if "总览" in request["task"]:
            first = request["candidates_by_section"]["ai-tech"][0]
            return json.dumps(
                {
                    "briefing": {
                        "title": "今日总结",
                        "summary": "今日信息围绕 AI、国际事务和市场变化展开，适合快速把握重点。",
                        "title_en": "Daily Briefing",
                        "summary_en": "Today focuses on AI, world affairs, and market moves.",
                    },
                    "headline": {
                        "title": "AI 框架更新",
                        "title_en": first["title"],
                        "url": first["url"],
                        "source_zh": first["source_zh"],
                        "source_en": first["source_en"],
                        "summary": "OpenAI 发布新的智能体框架，开发者工作流继续演进。",
                        "summary_en": first["summary"],
                        "reason": "代表今日 AI 工具进展",
                        "reason_en": "It represents today's AI tooling progress.",
                        "score": first["score"],
                    },
                }
            )

        section = request["section"]
        candidates = request["candidates"]
        subsections = []
        for subsection in section["subsections"]:
            chosen = next(
                (item for item in candidates if item["suggested_subsection"] == subsection["id"]),
                candidates[0] if candidates else None,
            )
            subsections.append(
                {
                    "id": subsection["id"],
                    "briefing": {
                        "title": f"{subsection['title']}总结",
                        "summary": f"{subsection['title']}今天有清晰线索可读。",
                        "title_en": f"{subsection['title_en']} Briefing",
                        "summary_en": f"{subsection['title_en']} has a clear item today.",
                    },
                    "articles": [
                        {
                            "title": f"{subsection['title']}新闻",
                            "title_en": chosen["title"],
                            "url": chosen["url"],
                            "source_zh": chosen["source_zh"],
                            "source_en": chosen["source_en"],
                            "summary": "这是一条面向读者的正式中文摘要。",
                            "summary_en": chosen["summary"],
                            "reason": "与本小版主题相关",
                            "reason_en": "Relevant to the subsection.",
                            "score": chosen["score"],
                        }
                    ]
                    if chosen
                    else [],
                }
            )
        return json.dumps(
            {
                "id": section["id"],
                "briefing": {
                    "title": f"{section['title']}总结",
                    "summary": f"{section['title']}版面完成编纂。",
                    "title_en": f"{section['title_en']} Briefing",
                    "summary_en": f"{section['title_en']} section is compiled.",
                },
                "subsections": subsections,
            }
        )

    monkeypatch.setattr("information_daily.compiler._post_chat_completion", fake_post_chat_completion)

    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=False)

    assert len(calls) == 1 + len(config.sections)
    assert "总览" in calls[0]
    assert [section.id for section in issue.sections] == [section.id for section in config.sections]
    assert issue.briefing_summary.startswith("今日信息围绕")
    assert issue.headline and issue.headline.source == "OpenAI 新闻"


def test_llm_max_input_items_limits_payload(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    config = replace(config, llm=replace(config.llm, max_input_items=2))
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    overview_request = {}

    def fake_post_chat_completion(config, payload, api_key):
        del config, api_key
        request = json.loads(payload["messages"][1]["content"])
        if "总览" in request["task"]:
            overview_request.update(request)
            first = request["candidates_by_section"]["ai-tech"][0]
            return json.dumps(
                {
                    "briefing": {
                        "title": "今日总结",
                        "summary": "今日信息围绕 AI、国际事务和市场变化展开。",
                        "title_en": "Daily Briefing",
                        "summary_en": "Today focuses on AI, world affairs, and market moves.",
                    },
                    "headline": {
                        "title": "AI 框架更新",
                        "title_en": first["title"],
                        "url": first["url"],
                        "source_zh": "错误来源",
                        "source_en": "Wrong Source",
                        "summary": "OpenAI 发布新的智能体框架。",
                        "summary_en": first["summary"],
                        "score": first["score"],
                    },
                }
            )
        section = request["section"]
        return json.dumps(
            {
                "id": section["id"],
                "briefing": {
                    "title": f"{section['title']}总结",
                    "summary": f"{section['title']}版面完成编纂。",
                    "title_en": f"{section['title_en']} Briefing",
                    "summary_en": f"{section['title_en']} section is compiled.",
                },
                "subsections": [],
            }
        )

    monkeypatch.setattr("information_daily.compiler._post_chat_completion", fake_post_chat_completion)

    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=False)

    total_candidates = sum(len(items) for items in overview_request["candidates_by_section"].values())
    assert total_candidates == 2
    assert issue.headline.original_title == "OpenAI releases a new agent framework"
    assert issue.headline.source == "OpenAI 新闻"
    assert issue.headline.source_en == "OpenAI News"


def test_llm_invalid_duplicate_and_missing_sections_are_repaired(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")

    def fake_post_chat_completion(config, payload, api_key):
        del config, api_key
        request = json.loads(payload["messages"][1]["content"])
        if "总览" in request["task"]:
            return json.dumps(
                {
                    "briefing": {
                        "title": "今日总结",
                        "summary": "今日信息围绕 AI、国际事务和市场变化展开。",
                    },
                    "headline": {
                        "title": "未知头条",
                        "url": "https://example.com/unknown",
                        "summary": "不存在的候选。",
                    },
                }
            )
        section = request["section"]
        if section["id"] != "ai-tech":
            return json.dumps({"id": section["id"], "subsections": []})
        first_subsection = section["subsections"][0]
        return json.dumps(
            {
                "id": "ai-tech",
                "briefing": {"title": "AI 科技总结", "summary": "AI 新闻。"},
                "subsections": [
                    {
                        "id": first_subsection["id"],
                        "briefing": {"title": "前沿模型总结", "summary": "模型新闻。"},
                        "articles": [
                            {
                                "title": "CLI 工具",
                                "url": "https://example.com/oss-cli",
                                "summary": "一款开源 CLI 工具。",
                            },
                            {
                                "title": "Duplicate CLI 工具",
                                "url": "https://example.com/oss-cli",
                                "summary": "重复链接。",
                            },
                            {
                                "title": "Unknown",
                                "url": "https://example.com/not-in-candidates",
                                "summary": "不存在的候选。",
                            },
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("information_daily.compiler._post_chat_completion", fake_post_chat_completion)

    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=False)

    warning_text = "\n".join(issue.warnings)
    assert "unknown URL" in warning_text
    assert "duplicate URL" in warning_text
    assert "missed subsection" in warning_text
    urls = [article.url for section in issue.sections for article in section.articles]
    assert len(urls) == len(set(urls))


def test_render_issue_writes_pages(tmp_path, monkeypatch):
    root = _copy_project_bits(tmp_path)
    config = load_config("ai-tech", root)
    monkeypatch.delenv(config.llm.api_key_env, raising=False)
    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=True)

    out_dir = tmp_path / "docs"
    render_issue(config, issue, out_dir, fetch_result=_fetch_result(issue, _articles()))

    index = (out_dir / "index.html").read_text(encoding="utf-8")
    en_index = (out_dir / "en" / "index.html").read_text(encoding="utf-8")
    archive = (out_dir / "archive.html").read_text(encoding="utf-8")
    en_archive = (out_dir / "en" / "archive.html").read_text(encoding="utf-8")
    sources = (out_dir / "sources.html").read_text(encoding="utf-8")
    raw = (out_dir / "raw" / "2026-05-29.html").read_text(encoding="utf-8")
    raw_json = json.loads((root / "data" / "raw" / "2026-05-29.json").read_text(encoding="utf-8"))
    ai_page = (out_dir / "sections" / "ai-tech.html").read_text(encoding="utf-8")
    en_ai_page = (out_dir / "en" / "sections" / "ai-tech.html").read_text(encoding="utf-8")
    paper = out_dir / "papers" / "2026-05-29.html"
    en_paper = out_dir / "en" / "papers" / "2026-05-29.html"
    dated_ai_page = out_dir / "papers" / "2026-05-29" / "ai-tech.html"
    en_dated_ai_page = out_dir / "en" / "papers" / "2026-05-29" / "ai-tech.html"

    assert "信息日报" in index
    assert "本期三版" in index
    assert "原始候选" in index
    assert "数据源状态" in index
    assert "sections/ai-tech.html" in index
    assert "en/index.html" in index
    assert "Information Daily" in en_index
    assert "Daily Editions" in en_index
    assert "../index.html" in en_index
    assert "前沿模型" in ai_page
    assert "开发与工具" in ai_page
    assert "研究与产品" in ai_page
    assert "前沿模型总结" in ai_page
    assert "Frontier Models" in en_ai_page
    assert "Developer Tools" in en_ai_page
    assert "Research &amp; Products" in en_ai_page
    assert index.count('class="sheet') == 1
    assert 'target="_blank" rel="noopener"' in index
    assert paper.exists()
    assert en_paper.exists()
    assert dated_ai_page.exists()
    assert en_dated_ai_page.exists()
    assert "往期信息日报" in archive
    assert "Information Daily Archive" in en_archive
    assert "OpenAI News" in sources
    assert "原始候选" in raw
    assert set(raw_json) == {"date", "profile_id", "generated_at", "sources", "articles", "warnings"}
    assert raw_json["articles"][0]["rank"] == 1


def _articles():
    return [
        Article(
            id="a1",
            title="OpenAI releases a new agent framework",
            url="https://example.com/openai-agent",
            source_id="openai-news",
            source_name="OpenAI News",
            default_section="ai-tech",
            summary="A short update about AI agents and developer workflows.",
            published_at=datetime(2026, 5, 29, 1, 0, tzinfo=timezone.utc),
            weight=1.2,
        ),
        Article(
            id="a2",
            title="Useful open source CLI for developers",
            url="https://example.com/oss-cli",
            source_id="github-blog",
            source_name="GitHub Blog",
            default_section="ai-tech",
            summary="A command line tool for open source maintainers.",
            published_at=datetime(2026, 5, 29, 2, 0, tzinfo=timezone.utc),
            weight=1.0,
        ),
        Article(
            id="a3",
            title="New LLM benchmark compares agent coding performance",
            url="https://example.com/llm-benchmark",
            source_id="huggingface-blog",
            source_name="Hugging Face Blog",
            default_section="ai-tech",
            summary="A benchmark compares how current AI models perform on coding agent tasks.",
            published_at=datetime(2026, 5, 29, 3, 0, tzinfo=timezone.utc),
            weight=0.9,
        ),
    ]


def _fetch_result(issue, articles):
    return FetchResult(
        articles=tuple(articles),
        warnings=(),
        sources=(
            SourceFetchStatus(
                id="openai-news",
                name="OpenAI News",
                type="rss",
                category="ai-tech",
                homepage="https://example.com/rss.xml",
                language="en",
                enabled=True,
                status="success",
                count=len(articles),
                duration_ms=12,
                error="",
                fetched_at=issue.generated_at.isoformat() if issue.generated_at else "",
            ),
        ),
    )


def _copy_project_bits(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    copytree(Path.cwd() / "config", root / "config")
    copytree(Path.cwd() / "templates", root / "templates")
    return root
