from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class SectionConfig:
    id: str
    title: str
    icon: str = ""
    max_articles: int = 6
    keywords: tuple[str, ...] = ()

    @property
    def display_title(self) -> str:
        return f"{self.icon} {self.title}".strip()


@dataclass(frozen=True)
class SourceConfig:
    id: str
    name: str
    type: str
    enabled: bool
    default_section: str
    weight: float = 1.0
    url: str | None = None
    keywords: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    provider: str
    api_key_env: str
    base_url_env: str
    model_env: str
    default_base_url: str
    default_model: str
    temperature: float
    max_input_items: int
    summary_style: str


@dataclass(frozen=True)
class AppConfig:
    root: Any
    profile_id: str
    site: dict[str, Any]
    sections: tuple[SectionConfig, ...]
    sources: tuple[SourceConfig, ...]
    llm: LLMConfig
    selection: dict[str, Any]

    @property
    def section_by_id(self) -> dict[str, SectionConfig]:
        return {section.id: section for section in self.sections}


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    url: str
    source_id: str
    source_name: str
    default_section: str
    summary: str = ""
    published_at: datetime | None = None
    weight: float = 1.0
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledArticle:
    title: str
    url: str
    source: str
    summary: str
    published_at: str | None = None
    reason: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class CompiledSection:
    id: str
    title: str
    icon: str
    articles: tuple[CompiledArticle, ...]

    @property
    def display_title(self) -> str:
        return f"{self.icon} {self.title}".strip()


@dataclass(frozen=True)
class Issue:
    date: date
    site_title: str
    site_subtitle: str
    edition_label: str
    profile_id: str
    briefing_title: str
    briefing_summary: str
    headline: CompiledArticle | None
    sections: tuple[CompiledSection, ...]
    source_count: int
    raw_count: int
    warnings: tuple[str, ...] = ()
    generated_at: datetime | None = None

    @property
    def slug(self) -> str:
        return self.date.isoformat()

    @property
    def title(self) -> str:
        return f"{self.site_title} {self.slug}"
