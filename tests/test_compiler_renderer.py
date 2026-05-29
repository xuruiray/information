from datetime import date, datetime, timezone
from pathlib import Path
from shutil import copytree

import pytest

from information_daily.compiler import compile_issue
from information_daily.config import ConfigError, load_config
from information_daily.models import Article
from information_daily.renderer import render_issue


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
    assert issue.warnings


def test_render_issue_writes_pages(tmp_path, monkeypatch):
    root = _copy_project_bits(tmp_path)
    config = load_config("ai-tech", root)
    monkeypatch.delenv(config.llm.api_key_env, raising=False)
    issue = compile_issue(config, _articles(), date(2026, 5, 29), allow_fallback=True)

    out_dir = tmp_path / "docs"
    render_issue(config, issue, out_dir)

    index = (out_dir / "index.html").read_text(encoding="utf-8")
    archive = (out_dir / "archive.html").read_text(encoding="utf-8")
    paper = out_dir / "papers" / "2026-05-29.html"

    assert "信息日报" in index
    assert "本期三版" in index
    assert "AI 科技总结" in index
    assert "国际事务总结" in index
    assert "财经理财总结" in index
    assert index.count('class="sheet') == 3
    assert 'target="_blank" rel="noopener"' in index
    assert paper.exists()
    assert "往期信息日报" in archive


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


def _copy_project_bits(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    copytree(Path.cwd() / "config", root / "config")
    copytree(Path.cwd() / "templates", root / "templates")
    return root
