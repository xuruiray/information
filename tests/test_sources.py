from pathlib import Path
from datetime import datetime, timezone

from information_daily.models import AppConfig, Article, LLMConfig, SectionConfig, SourceConfig
from information_daily.sources import SourceError, fetch_all, fetch_source


def test_parse_rss_fixture(monkeypatch):
    payload = (Path(__file__).parent / "fixtures" / "sample.rss").read_bytes()
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Response(payload))

    articles = fetch_source(
        SourceConfig(
            id="sample",
            name="Sample",
            type="rss",
            enabled=True,
            url="https://example.com/rss.xml",
            default_section="ai",
        )
    )

    assert len(articles) == 2
    assert articles[0].title == "OpenAI releases a new agent framework"
    assert articles[0].url == "https://example.com/openai-agent"
    assert articles[0].summary == "A short update about AI agents and developer workflows."


def test_parse_atom_fixture(monkeypatch):
    payload = (Path(__file__).parent / "fixtures" / "sample.atom").read_bytes()
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Response(payload))

    articles = fetch_source(
        SourceConfig(
            id="sample",
            name="Sample",
            type="atom",
            enabled=True,
            url="https://example.com/feed",
            default_section="open-source",
        )
    )

    assert len(articles) == 1
    assert articles[0].title == "Launch of a useful open source CLI"
    assert articles[0].published_at is not None


def test_x_source_is_reserved_placeholder():
    articles = fetch_source(
        SourceConfig(
            id="x-ai",
            name="X AI",
            type="x",
            enabled=True,
            default_section="ai",
            options={"token_env": "X_BEARER_TOKEN"},
        )
    )

    assert articles == []


def test_fetch_all_collects_source_statuses(monkeypatch):
    config = AppConfig(
        root=Path.cwd(),
        profile_id="test",
        site={},
        sections=(SectionConfig(id="ai", title="AI"),),
        sources=(
            SourceConfig(
                id="good",
                name="Good",
                type="rss",
                enabled=True,
                default_section="ai",
                category="ai",
            ),
            SourceConfig(
                id="bad",
                name="Bad",
                type="rss",
                enabled=True,
                default_section="ai",
                category="ai",
            ),
            SourceConfig(
                id="off",
                name="Off",
                type="rss",
                enabled=False,
                default_section="ai",
                category="ai",
            ),
        ),
        llm=LLMConfig(
            enabled=False,
            provider="test",
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
            model_env="OPENAI_MODEL",
            default_base_url="https://example.com",
            default_model="test",
            temperature=0,
            max_input_items=10,
            summary_style="",
        ),
        selection={"max_per_source": 2},
    )

    def fake_fetch_source(source, timeout):
        if source.id == "bad":
            raise SourceError("Source bad returned HTTP 500")
        return [
            Article(
                id="a1",
                title="One",
                url="https://example.com/one",
                source_id=source.id,
                source_name=source.name,
                default_section="ai",
                published_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
            )
        ]

    monkeypatch.setattr("information_daily.sources.fetch_source", fake_fetch_source)

    result = fetch_all(config, timeout=1, workers=2)

    assert [article.source_id for article in result.articles] == ["good"]
    assert result.warnings == ("Source bad returned HTTP 500",)
    statuses = {status.id: status for status in result.sources}
    assert statuses["good"].status == "success"
    assert statuses["good"].count == 1
    assert statuses["bad"].status == "error"
    assert statuses["off"].status == "disabled"


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload
