from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from time import perf_counter
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import AppConfig, Article, SourceConfig
from .utils import canonical_url, normalize_space, parse_datetime, stable_id, strip_html


class SourceError(RuntimeError):
    """Raised when a source cannot be fetched or parsed."""


@dataclass(frozen=True)
class FetchResult:
    articles: tuple[Article, ...]
    warnings: tuple[str, ...]
    sources: tuple[SourceFetchStatus, ...] = ()


@dataclass(frozen=True)
class SourceFetchStatus:
    id: str
    name: str
    type: str
    category: str
    homepage: str
    language: str
    enabled: bool
    status: str
    count: int
    duration_ms: int
    error: str
    fetched_at: str


def fetch_all(config: AppConfig, timeout: int = 20, workers: int = 8) -> FetchResult:
    max_per_source = int(config.selection.get("max_per_source") or 20)
    enabled_sources = [source for source in config.sources if source.enabled]
    source_articles: dict[str, tuple[Article, ...]] = {}
    source_statuses: dict[str, SourceFetchStatus] = {
        source.id: _source_status(source, status="disabled")
        for source in config.sources
        if not source.enabled
    }
    warnings: list[str] = []

    if enabled_sources:
        max_workers = max(1, min(int(workers or 1), len(enabled_sources)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_source = {
                executor.submit(_fetch_enabled_source, source, timeout, max_per_source): source
                for source in enabled_sources
            }
            for future in as_completed(future_by_source):
                source = future_by_source[future]
                articles, status, warning = future.result()
                source_articles[source.id] = articles
                source_statuses[source.id] = status
                if warning:
                    warnings.append(warning)

    articles: list[Article] = []
    statuses: list[SourceFetchStatus] = []
    for source in config.sources:
        status = source_statuses.get(source.id) or _source_status(
            source,
            status="error",
            error=f"Source {source.id} did not complete",
        )
        statuses.append(status)
        articles.extend(source_articles.get(source.id, ()))
    return FetchResult(
        articles=tuple(articles),
        warnings=tuple(warnings),
        sources=tuple(statuses),
    )


def _fetch_enabled_source(
    source: SourceConfig,
    timeout: int,
    max_per_source: int,
) -> tuple[tuple[Article, ...], SourceFetchStatus, str]:
    started = perf_counter()
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        articles = tuple(fetch_source(source, timeout=timeout)[:max_per_source])
    except SourceError as exc:
        warning = str(exc)
        return (
            (),
            _source_status(
                source,
                status="error",
                duration_ms=_elapsed_ms(started),
                error=warning,
                fetched_at=fetched_at,
            ),
            warning,
        )
    except Exception as exc:
        warning = f"Source {source.id} failed unexpectedly: {exc}"
        return (
            (),
            _source_status(
                source,
                status="error",
                duration_ms=_elapsed_ms(started),
                error=warning,
                fetched_at=fetched_at,
            ),
            warning,
        )
    return (
        articles,
        _source_status(
            source,
            status="success",
            count=len(articles),
            duration_ms=_elapsed_ms(started),
            fetched_at=fetched_at,
        ),
        "",
    )


def _source_status(
    source: SourceConfig,
    status: str,
    count: int = 0,
    duration_ms: int = 0,
    error: str = "",
    fetched_at: str = "",
) -> SourceFetchStatus:
    return SourceFetchStatus(
        id=source.id,
        name=source.name,
        type=source.type,
        category=source.category or source.default_section,
        homepage=source.homepage or source.url or "",
        language=source.language,
        enabled=source.enabled,
        status=status,
        count=count,
        duration_ms=duration_ms,
        error=error,
        fetched_at=fetched_at,
    )


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


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
