import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from shutil import copytree

import pytest

from information_daily.compiler import (
    CompileError,
    _issue_from_llm_data,
    _best_subsection,
    _score_article_for_section,
    compile_issue,
)
from information_daily.config import ConfigError, load_config
from information_daily.models import Article
from information_daily.renderer import render_issue
from information_daily.sources import FetchResult, SourceFetchStatus
from information_daily.cli import _load_source_snapshot


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
    assert [section.id for section in issue.sections] == [
        section.id for section in config.sections
    ]
    assert any(
        section.articles for section in issue.sections if section.id == "ai-tech"
    )
    assert all(section.briefing_summary for section in issue.sections)
    assert all(len(section.subsections) == 3 for section in issue.sections)
    assert all(
        subsection.briefing_summary
        for section in issue.sections
        for subsection in section.subsections
    )
    ai_articles = next(
        section.articles for section in issue.sections if section.id == "ai-tech"
    )
    assert ai_articles[0].title == ai_articles[0].original_title
    assert ai_articles[0].summary in {a.summary for a in _articles()}
    assert "关注的关键线索" not in ai_articles[0].title
    assert issue.warnings


def test_sparse_fallback_leaves_unrelated_subsections_empty(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.delenv(config.llm.api_key_env, raising=False)
    article = replace(
        _articles()[0], title="OpenAI launches an agent", summary="An AI agent update."
    )
    issue = compile_issue(config, [article], date(2026, 5, 29), allow_fallback=True)
    for section in issue.sections:
        if section.id != "ai-tech":
            assert not section.articles
    ai = next(section for section in issue.sections if section.id == "ai-tech")
    assert len(ai.articles) == 1
    assert not next(sub for sub in ai.subsections if sub.id == "ai-products").articles
    assert ai.articles[0].title in ai.briefing_summary


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


def _edited(article, **changes):
    return {
        "title": "OpenAI 发布智能体框架",
        "summary": "OpenAI 发布新的智能体框架，支持开发者工作流。",
        "title_en": article.title,
        "summary_en": article.summary,
        "url": article.url,
        "topic_evidence": "agent",
        **changes,
    }


def _briefing(articles):
    return {
        "title": "今日总结",
        "summary": "；".join(a["title"] for a in articles) + "。",
        "title_en": "Daily Briefing",
        "summary_en": "; ".join(a["title_en"] for a in articles),
        "covered_urls": [a["url"] for a in articles],
    }


def _fake_editor(calls, selected, mutate=None):
    def post(config, payload, api_key):
        request = json.loads(payload["messages"][1]["content"])
        calls.append(request)
        if "final_sections" in request:
            articles = [
                a for section in request["final_sections"] for a in section["articles"]
            ]
            response = {
                "briefing": _briefing(articles),
                "headline_url": articles[0]["url"],
            }
        elif "final_subsections" in request:
            articles = [
                a for sub in request["final_subsections"] for a in sub["articles"]
            ]
            response = {
                "briefing": _briefing(articles),
                "subsections": [
                    {"id": sub["id"], "briefing": _briefing(sub["articles"])}
                    for sub in request["final_subsections"]
                ],
            }
        else:
            section_id = request["section"]["id"]
            response = {
                "id": section_id,
                "briefing": {"summary": "不应保留的旧总结。"},
                "subsections": selected.get(section_id, []),
            }
        if mutate:
            mutate(request, response)
        return json.dumps(response)

    return post


def test_llm_summaries_only_receive_final_articles_after_deduplication(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    article = _edited(_articles()[0])
    calls = []
    chosen = {
        "ai-tech": [
            {
                "id": "frontier-ai",
                "articles": [article, article],
                "briefing": {"summary": "暂无文章入选。"},
            }
        ]
    }
    monkeypatch.setattr(
        "information_daily.compiler._post_chat_completion", _fake_editor(calls, chosen)
    )
    issue = compile_issue(config, _articles(), date(2026, 5, 29))
    ai = next(s for s in issue.sections if s.id == "ai-tech")
    assert len(ai.articles) == 1
    assert not ai.subsections[1].articles
    assert ai.briefing_summary == article["title"] + "。"
    assert ai.subsections[0].briefing_summary == article["title"] + "。"
    assert issue.headline == ai.articles[0]
    assert "不应保留" not in issue.briefing_summary
    assert any("duplicate URL" in w for w in issue.warnings)
    final_calls = [
        c for c in calls if "final_subsections" in c or "final_sections" in c
    ]
    assert len(final_calls) == 2
    assert calls[-1] == final_calls[-1]
    assert (
        final_calls[0]["final_subsections"][0]["articles"][0]["url"] == article["url"]
    )
    assert len(final_calls[1]["final_sections"][0]["articles"]) == 1


def test_invalid_copy_unknown_urls_and_wrong_subsections_are_skipped_without_refill():
    config = load_config("ai-tech", Path.cwd())
    good = _edited(_articles()[0])
    generic = _edited(_articles()[2], title="AI 与大模型：某来源关注的关键线索")
    missing = _edited(_articles()[2], summary="")
    unknown = _edited(_articles()[0], url="https://example.com/not-a-candidate")
    cli = _edited(_articles()[2], title="模型评测报告", topic_evidence="benchmark")
    data = {
        "sections": [
            {
                "id": "ai-tech",
                "subsections": [
                    {
                        "id": "frontier-ai",
                        "articles": [good, generic, missing, unknown],
                    },
                    {"id": "ai-products", "articles": [cli]},
                ],
            }
        ]
    }
    issue = _issue_from_llm_data(config, data, _articles(), date(2026, 5, 29))
    ai = issue.sections[0]
    assert [a.url for a in ai.articles] == [good["url"]]
    assert not ai.subsections[1].articles
    assert all(not s.articles for s in issue.sections[1:])
    warnings = "\n".join(issue.warnings)
    for expected in [
        "generic template",
        "missing concrete",
        "unknown URL",
        "topic evidence does not match",
    ]:
        assert expected in warnings


@pytest.mark.parametrize("evidence", ["", "a made-up agent result", "arXiv"])
def test_missing_fabricated_and_metadata_only_topic_evidence_is_rejected(evidence):
    config = load_config("ai-tech", Path.cwd())
    candidate = replace(_articles()[0], summary="arXiv abstract: an agent update.")
    data = {
        "sections": [
            {
                "id": "ai-tech",
                "subsections": [
                    {
                        "id": "ai-research-governance",
                        "articles": [_edited(candidate, topic_evidence=evidence)],
                    }
                ],
            }
        ]
    }
    issue = _issue_from_llm_data(config, data, [candidate], date(2026, 5, 29))
    assert not any(section.articles for section in issue.sections)
    assert issue.warnings


def test_section_candidate_whitelist_blocks_cross_section_news():
    config = load_config("ai-tech", Path.cwd())
    war = replace(
        _articles()[0],
        id="war",
        title="Trump weighs joining war against Houthis",
        summary="Government discusses diplomacy and war.",
        url="https://example.com/war",
        default_section="world",
    )
    bogus = _edited(war, title="美国讨论加入战争", topic_evidence="war")
    issue = _issue_from_llm_data(
        config,
        {
            "sections": [
                {
                    "id": "developer",
                    "subsections": [{"id": "infra-data", "articles": [bogus]}],
                }
            ]
        },
        [war],
        date(2026, 5, 29),
    )
    assert not any(s.articles for s in issue.sections)
    assert any("unknown URL" in w for w in issue.warnings)


@pytest.mark.parametrize(
    "title,section_id,subsection_id,evidence",
    [
        (
            "Scientists discover vanilla’s hidden healing potential",
            "science-tech",
            "science-research",
            "vanillin, the compound that gives vanilla its distinctive flavor",
        ),
        (
            "国家铁路局发布《铁路科技创新“十五五”规划》",
            "china",
            "china-policy",
            "铁路科技创新“十五五”规划",
        ),
        (
            "江西省政协原副主席胡幼桃受贿案一审被判十三年",
            "china",
            "china-society-law",
            "胡幼桃受贿案一审",
        ),
        (
            "The gap in shared understanding of LLM capability is widening",
            "opinion",
            "tech-opinion",
            "LLM capability",
        ),
        (
            "Passkeys are a trap",
            "opinion",
            "community-discussion",
            "Passkeys are a trap",
        ),
    ],
)
def test_saved_rss_events_are_not_rejected_for_a_keyword_free_quote(
    title, section_id, subsection_id, evidence
):
    config = load_config("ai-tech", Path.cwd())
    fetch = _load_source_snapshot(
        Path("tests/fixtures/editorial-audit-2026-10-12.json"),
        "ai-tech",
        date(2026, 10, 12),
    )
    candidate = next(a for a in fetch.articles if a.title == title)
    decisions = []
    issue = _issue_from_llm_data(
        config,
        {
            "sections": [
                {
                    "id": section_id,
                    "subsections": [
                        {
                            "id": subsection_id,
                            "articles": [
                                _edited(
                                    candidate,
                                    title="原文事件的中文审阅标题",
                                    summary="根据保存的原始素材检验主题证据规则。",
                                    topic_evidence=evidence,
                                )
                            ],
                        }
                    ],
                }
            ]
        },
        [candidate],
        date(2026, 10, 12),
        decisions=decisions,
    )
    assert [a.url for s in issue.sections for a in s.articles] == [candidate.url]
    assert decisions[0]["status"] == "accepted"
    assert decisions[0]["original_title"] == title
    assert decisions[0]["source_keyword_matches"]


def test_quote_normalization_allows_typography_but_not_paraphrases():
    from information_daily.compiler import _topic_evidence_warning

    config = load_config("ai-tech", Path.cwd())
    sub = config.sections[0].subsections[0]
    candidate = replace(_articles()[0], title="An AI model’s new result", summary="")
    assert not _topic_evidence_warning(sub, candidate, "AI model's new result")
    assert _topic_evidence_warning(sub, candidate, "AI model delivers a new result")


def test_replaying_legacy_sources_does_not_inherit_previous_editor_rejections():
    fetch = _load_source_snapshot(
        Path("tests/fixtures/editorial-audit-2026-10-12.json"),
        "ai-tech",
        date(2026, 10, 12),
    )
    assert not fetch.warnings
    assert len(fetch.articles) == 371
    assert len(fetch.sources) == 33


def test_editorial_audit_preserves_rejected_url_and_raw_selection(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    good = _edited(_articles()[0])
    bad = _edited(_articles()[2], topic_evidence="invented phrase")
    selected = {"ai-tech": [{"id": "frontier-ai", "articles": [good, bad]}]}
    monkeypatch.setattr(
        "information_daily.compiler._post_chat_completion", _fake_editor([], selected)
    )
    audit = {}
    issue = compile_issue(config, _articles(), date(2026, 5, 29), editorial_audit=audit)
    assert audit["sections"][0]["subsections"][0]["articles"][1] == bad
    rejected = next(d for d in audit["decisions"] if d["status"] == "rejected")
    assert rejected["url"] == bad["url"]
    assert rejected["reason"] == "missing verbatim topic evidence"
    assert rejected["url"] in issue.warnings[0]


@pytest.mark.parametrize("approved", [True, False])
def test_keyword_misses_receive_grounded_review_before_final_summaries(
    monkeypatch, approved
):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    candidate = replace(
        _articles()[0],
        id="nadella",
        url="https://example.com/nadella",
        title="Microsoft’s Satya Nadella says AI models need an ‘emergency brake’",
        summary="Microsoft's CEO wrote that it’s time to assess the trust architecture of AI.",
    )
    saved = {"sections": [{"id": s.id, "subsections": []} for s in config.sections]}
    saved["sections"][0]["subsections"] = [
        {"id": "frontier-ai", "articles": [_edited(_articles()[0])]},
        {
            "id": "ai-research-governance",
            "articles": [
                _edited(
                    candidate,
                    title="纳德拉呼吁AI模型设置紧急制动",
                    topic_evidence="AI models need an ‘emergency brake’",
                )
            ],
        },
    ]
    calls = []
    summarize = _fake_editor(calls, {})

    def post(config, payload, key):
        request = json.loads(payload["messages"][1]["content"])
        if "topic_review_candidates" not in request:
            return summarize(config, payload, key)
        calls.append(request)
        review = request["topic_review_candidates"][0]
        assert review["original_title"] == candidate.title
        return json.dumps(
            {
                "decisions": [
                    {
                        "id": review["id"],
                        "approved": approved,
                        "evidence": "AI models need an ‘emergency brake’",
                        "reason": "AI紧急制动涉及模型安全与治理。",
                    }
                ]
            }
        )

    monkeypatch.setattr("information_daily.compiler._post_chat_completion", post)
    audit = {}
    issue = compile_issue(
        config,
        [_articles()[0], candidate],
        date(2026, 5, 29),
        editorial_selection=saved,
        editorial_audit=audit,
    )
    published = {a.url for s in issue.sections for a in s.articles}
    assert (candidate.url in published) == approved
    assert len(calls) == 3  # one review, final section summary, final overview
    assert audit["topic_review"][0]["approved"] == approved
    assert len(calls[-1]["final_sections"][0]["articles"]) == (2 if approved else 1)


@pytest.mark.parametrize("change", ["invented_evidence", "unknown_id"])
def test_topic_review_cannot_approve_untraceable_source_evidence(change):
    from information_daily.compiler import _validated_topic_review

    candidate = {
        "id": "ai-tech/ai-research-governance/url",
        "url": "https://example.com",
        "original_title": "AI models need an emergency brake",
        "original_summary": "",
    }
    decision = {
        "id": candidate["id"],
        "approved": True,
        "evidence": candidate["original_title"],
        "reason": "AI安全治理",
    }
    if change == "invented_evidence":
        decision["evidence"] = "The company released new safety legislation"
    else:
        decision["id"] = "another input"
    with pytest.raises(CompileError):
        _validated_topic_review({"decisions": [decision]}, [candidate])


def test_punctuation_around_quotes_is_allowed_without_allowing_rewritten_words():
    from information_daily.compiler import _verbatim_evidence_warning

    original = replace(
        _articles()[0],
        summary="Oil swung after an “energy ceasefire,” while strikes continued.",
    )
    assert not _verbatim_evidence_warning(
        original, "Oil swung after an “energy ceasefire”"
    )
    assert _verbatim_evidence_warning(original, "Oil fell after an “energy ceasefire”")
    assert _verbatim_evidence_warning(original, "...")


def test_short_keyword_and_source_tags_do_not_establish_topic():
    config = load_config("ai-tech", Path.cwd())
    article = replace(
        _articles()[0],
        title="Riyadh airport attack",
        summary="Details of the attack.",
        default_section="world",
        keywords=("go", "ai", "database"),
    )
    developer = config.section_by_id["developer"]
    ai = config.section_by_id["ai-tech"]
    assert _score_article_for_section(developer, article) == article.weight
    assert _score_article_for_section(ai, article) == article.weight
    article = replace(
        article, title="Python release for developers", summary="Python version update."
    )
    assert _best_subsection(developer, article).id == "languages-frameworks"


@pytest.mark.parametrize(
    "failure",
    ["urls", "empty_summary", "contradiction", "no_selected", "unknown_headline"],
)
def test_bad_final_summaries_fail_closed_after_one_retry(monkeypatch, failure):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    selected = {
        "ai-tech": [{"id": "frontier-ai", "articles": [_edited(_articles()[0])]}]
    }
    calls = []

    def corrupt(request, response):
        if "final_subsections" in request:
            if failure == "urls":
                response["briefing"]["covered_urls"] = ["https://example.com/removed"]
            if failure == "empty_summary":
                response["briefing"]["summary"] = ""
            if failure == "contradiction":
                response["briefing"]["summary"] = "本栏目暂无文章入选。"
            if failure == "no_selected":
                response["briefing"]["summary"] = "本栏目暂无入选文章。"
        if "final_sections" in request and failure == "unknown_headline":
            response["headline_url"] = "https://example.com/not-published"

    monkeypatch.setattr(
        "information_daily.compiler._post_chat_completion",
        _fake_editor(calls, selected, corrupt),
    )
    with pytest.raises(CompileError):
        compile_issue(config, _articles(), date(2026, 5, 29))
    final_calls = [
        c
        for c in calls
        if ("final_sections" if failure == "unknown_headline" else "final_subsections")
        in c
    ]
    assert len(final_calls) == 2
    assert "repair" in final_calls[1]


def test_sparse_subsection_honors_both_upper_limits_and_has_no_minimum():
    config = load_config("ai-tech", Path.cwd())
    ai = replace(config.sections[0], max_articles=1)
    config = replace(config, sections=(ai, *config.sections[1:]))
    data = {
        "sections": [
            {
                "id": "ai-tech",
                "subsections": [
                    {
                        "id": "frontier-ai",
                        "articles": [_edited(_articles()[0]), _edited(_articles()[2])],
                    },
                ],
            }
        ]
    }
    issue = _issue_from_llm_data(config, data, _articles(), date(2026, 5, 29))
    assert sum(len(s.articles) for s in issue.sections) == 1
    assert len(issue.sections[0].subsections[0].articles) == 1


def test_llm_candidate_limit_and_empty_selection_do_not_trigger_raw_fallback(
    monkeypatch,
):
    config = load_config("ai-tech", Path.cwd())
    config = replace(config, llm=replace(config.llm, max_input_items=1))
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    calls = []
    monkeypatch.setattr(
        "information_daily.compiler._post_chat_completion", _fake_editor(calls, {})
    )
    with pytest.raises(CompileError, match="No publishable articles"):
        compile_issue(config, _articles(), date(2026, 5, 29))
    assert all(len(call["candidates"]) <= 1 for call in calls)


def test_render_issue_writes_pages(tmp_path, monkeypatch):
    root = _copy_project_bits(tmp_path)
    config = load_config("ai-tech", root)
    monkeypatch.delenv(config.llm.api_key_env, raising=False)
    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=True)
    hidden_warning = "internal duplicate URL diagnostic should stay out of reader pages"
    issue = replace(issue, warnings=(hidden_warning, *issue.warnings))

    out_dir = tmp_path / "docs"
    render_issue(config, issue, out_dir, fetch_result=_fetch_result(issue, _articles()))

    index = (out_dir / "index.html").read_text(encoding="utf-8")
    en_index = (out_dir / "en" / "index.html").read_text(encoding="utf-8")
    archive = (out_dir / "archive.html").read_text(encoding="utf-8")
    en_archive = (out_dir / "en" / "archive.html").read_text(encoding="utf-8")
    sources = (out_dir / "sources.html").read_text(encoding="utf-8")
    raw = (out_dir / "raw" / "2026-05-29.html").read_text(encoding="utf-8")
    raw_json = json.loads(
        (root / "data" / "raw" / "2026-05-29.json").read_text(encoding="utf-8")
    )
    ai_page = (out_dir / "sections" / "ai-tech.html").read_text(encoding="utf-8")
    en_ai_page = (out_dir / "en" / "sections" / "ai-tech.html").read_text(
        encoding="utf-8"
    )
    paper = out_dir / "papers" / "2026-05-29.html"
    en_paper = out_dir / "en" / "papers" / "2026-05-29.html"
    dated_ai_page = out_dir / "papers" / "2026-05-29" / "ai-tech.html"
    en_dated_ai_page = out_dir / "en" / "papers" / "2026-05-29" / "ai-tech.html"

    assert "信息日报" in index
    assert hidden_warning not in index
    assert hidden_warning not in en_index
    assert "本期版面" in index
    assert "原始候选" in index
    assert "数据源状态" in index
    assert "sections/ai-tech.html" in index
    assert "en/index.html" in index
    assert "Information Daily" in en_index
    assert "Daily Editions" in en_index
    assert "../index.html" in en_index
    assert "前沿模型" in ai_page
    assert "AI 产品与平台" in ai_page
    assert "研究与治理" in ai_page
    assert "前沿模型总结" in ai_page
    assert "Frontier Models" in en_ai_page
    assert "AI Products &amp; Platforms" in en_ai_page
    assert "Research &amp; Governance" in en_ai_page
    assert index.count('class="sheet') == 1
    assert 'target="_blank" rel="noopener"' in index
    assert paper.exists()
    assert en_paper.exists()
    assert hidden_warning not in paper.read_text(encoding="utf-8")
    assert hidden_warning not in en_paper.read_text(encoding="utf-8")
    assert dated_ai_page.exists()
    assert en_dated_ai_page.exists()
    assert "往期信息日报" in archive
    assert "Information Daily Archive" in en_archive
    assert "OpenAI News" in sources
    assert "原始候选" in raw
    assert set(raw_json) == {
        "date",
        "profile_id",
        "generated_at",
        "sources",
        "articles",
        "warnings",
        "fetch_warnings",
    }
    assert raw_json["fetch_warnings"] == []
    assert hidden_warning in raw_json["warnings"]
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


def test_malformed_selection_is_retried_before_final_summaries(monkeypatch):
    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.api_key_env, "test-key")
    selected = {
        "ai-tech": [{"id": "frontier-ai", "articles": [_edited(_articles()[0])]}]
    }
    calls = []
    valid_editor = _fake_editor(calls, selected)
    attempts = []

    def flaky_editor(config, payload, api_key):
        request = json.loads(payload["messages"][1]["content"])
        attempts.append(request)
        if len(attempts) == 1:
            return '{"id":"ai-tech","subsections":['
        return valid_editor(config, payload, api_key)

    monkeypatch.setattr(
        "information_daily.compiler._post_chat_completion", flaky_editor
    )
    issue = compile_issue(config, _articles(), date(2026, 5, 29))
    assert len(issue.sections[0].articles) == 1
    assert "repair" in attempts[1]
    assert attempts[0]["candidates"] == attempts[1]["candidates"]
    assert "final_sections" in attempts[-1]


def test_truncated_model_response_cannot_publish_partial_valid_json(monkeypatch):
    import io
    from information_daily.compiler import _request_editor

    config = load_config("ai-tech", Path.cwd())
    calls = []

    def truncated_response(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"id":"ai-tech","subsections":[]}'},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(
        "information_daily.compiler.urllib.request.urlopen", truncated_response
    )
    with pytest.raises(CompileError, match="finish_reason=length"):
        _request_editor(config, {"task": "Test incomplete model response"}, "test-key")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "model,budget_key",
    [
        ("gpt-5.5", "max_completion_tokens"),
        ("openai/gpt-6-sol", "max_completion_tokens"),
        ("gemini-2.5-pro", "max_tokens"),
    ],
)
def test_reasoning_model_budget_accounts_for_thinking(monkeypatch, model, budget_key):
    from information_daily.compiler import _llm_payload

    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.model_env, model)
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    payload = _llm_payload(config, [])
    assert payload["reasoning_effort"] == "low"
    assert payload[budget_key] == 16000
    if budget_key == "max_completion_tokens":
        assert "max_tokens" not in payload
        assert "temperature" not in payload
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "22000")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    overridden = _llm_payload(config, [])
    assert overridden[budget_key] == 22000
    assert overridden["reasoning_effort"] == "medium"


def test_nonreasoning_models_keep_original_provider_defaults(monkeypatch):
    from information_daily.compiler import _llm_payload

    config = load_config("ai-tech", Path.cwd())
    monkeypatch.setenv(config.llm.model_env, "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
    payload = _llm_payload(config, [])
    assert "reasoning_effort" not in payload
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload
    assert payload["temperature"] == config.llm.temperature
