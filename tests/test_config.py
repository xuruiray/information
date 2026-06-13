from pathlib import Path

from information_daily.config import load_config


def test_load_default_config():
    config = load_config("ai-tech", Path.cwd())

    assert config.profile_id == "ai-tech"
    assert config.site["title"] == "信息日报"
    assert [section.id for section in config.sections] == ["ai-tech", "world", "finance"]
    assert all(len(section.subsections) == 3 for section in config.sections)
    assert any(source.type == "x" for source in config.sources)
    assert any(source.id == "openai-news" for source in config.sources)
    assert any(source.id == "bbc-world" for source in config.sources)
    assert any(source.id == "bloomberg-markets" for source in config.sources)
    openai = next(source for source in config.sources if source.id == "openai-news")
    assert openai.category == "ai-tech"
    assert openai.homepage == "https://openai.com/news/rss.xml"
    assert openai.display is True
