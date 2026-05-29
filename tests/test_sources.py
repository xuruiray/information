from pathlib import Path

from information_daily.models import SourceConfig
from information_daily.sources import fetch_source


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


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload
