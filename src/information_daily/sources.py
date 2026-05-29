from __future__ import annotations

import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

from .models import AppConfig, Article, SourceConfig
from .utils import canonical_url, normalize_space, parse_datetime, stable_id, strip_html


class SourceError(RuntimeError):
    """Raised when a source cannot be fetched or parsed."""


@dataclass(frozen=True)
class FetchResult:
    articles: tuple[Article, ...]
    warnings: tuple[str, ...]


def fetch_all(config: AppConfig, timeout: int = 20) -> FetchResult:
    articles: list[Article] = []
    warnings: list[str] = []
    max_per_source = int(config.selection.get("max_per_source") or 20)

    for source in config.sources:
        if not source.enabled:
            continue
        try:
            source_articles = fetch_source(source, timeout=timeout)[:max_per_source]
        except SourceError as exc:
            warnings.append(str(exc))
            continue
        articles.extend(source_articles)
    return FetchResult(articles=tuple(articles), warnings=tuple(warnings))


def fetch_source(source: SourceConfig, timeout: int = 20) -> list[Article]:
    if source.type == "x":
        return _fetch_x_placeholder(source)
    if source.type not in {"rss", "atom"}:
        raise SourceError(f"Unsupported source type {source.type!r} for {source.id}")
    if not source.url:
        raise SourceError(f"Source {source.id} is missing url")

    request = urllib.request.Request(
        source.url,
        headers={
            "User-Agent": "information-daily/0.1 (+https://github.com/xuruiray/information)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise SourceError(f"Source {source.id} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SourceError(f"Source {source.id} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SourceError(f"Source {source.id} timed out") from exc

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise SourceError(f"Source {source.id} returned invalid XML") from exc

    if _local_name(root.tag) == "rss":
        return _parse_rss(source, root)
    return _parse_atom(source, root)


def _fetch_x_placeholder(source: SourceConfig) -> list[Article]:
    token_env = source.options.get("token_env") or "X_BEARER_TOKEN"
    print(
        f"Skipping X source {source.id}: adapter is reserved; future implementation will use {token_env}.",
        file=sys.stderr,
    )
    return []


def _parse_rss(source: SourceConfig, root: ET.Element) -> list[Article]:
    channel = _first_child(root, "channel") or root
    items = [child for child in channel if _local_name(child.tag) == "item"]
    articles = []
    for item in items:
        title = normalize_space(_child_text(item, "title"))
        link = normalize_space(_child_text(item, "link"))
        if not title or not link:
            continue
        summary = strip_html(_child_text(item, "description") or _child_text(item, "summary"))
        published = parse_datetime(
            _child_text(item, "pubDate")
            or _child_text(item, "published")
            or _child_text(item, "updated")
            or _child_text(item, "date")
        )
        articles.append(_article(source, title, link, summary, published))
    return articles


def _parse_atom(source: SourceConfig, root: ET.Element) -> list[Article]:
    entries = [child for child in root if _local_name(child.tag) == "entry"]
    articles = []
    for entry in entries:
        title = normalize_space(_child_text(entry, "title"))
        link = _atom_link(entry)
        if not title or not link:
            continue
        summary = strip_html(
            _child_text(entry, "summary")
            or _child_text(entry, "content")
            or _child_text(entry, "subtitle")
        )
        published = parse_datetime(_child_text(entry, "published") or _child_text(entry, "updated"))
        articles.append(_article(source, title, link, summary, published))
    return articles


def _article(
    source: SourceConfig,
    title: str,
    link: str,
    summary: str,
    published: datetime | None,
) -> Article:
    url = canonical_url(link)
    return Article(
        id=stable_id(url or title),
        title=title,
        url=url,
        source_id=source.id,
        source_name=source.name,
        default_section=source.default_section,
        summary=summary,
        published_at=published,
        weight=source.weight,
        keywords=source.keywords,
    )


def _atom_link(entry: ET.Element) -> str:
    fallback = ""
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if not href:
            continue
        if child.attrib.get("rel", "alternate") == "alternate":
            return href
        fallback = fallback or href
    return fallback


def _child_text(element: ET.Element, name: str) -> str:
    child = _first_child(element, name)
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag
