from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import AppConfig, LLMConfig, SectionConfig, SourceConfig, SubsectionConfig


class ConfigError(RuntimeError):
    """Raised when the YAML configuration is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def load_config(profile: str = "ai-tech", root: Path | None = None) -> AppConfig:
    root = (root or Path.cwd()).resolve()
    site_doc = load_yaml(root / "config" / "site.yaml")
    profile_doc = load_yaml(root / "config" / "profiles" / f"{profile}.yaml")

    site = _required_mapping(site_doc, "site", "config/site.yaml")
    profile_cfg = _required_mapping(profile_doc, "profile", f"config/profiles/{profile}.yaml")
    profile_id = str(profile_cfg.get("id") or profile)

    sections = tuple(_parse_section(item) for item in _required_list(profile_cfg, "sections"))
    if not sections:
        raise ConfigError("Profile must define at least one section")
    section_ids = {section.id for section in sections}
    if len(section_ids) != len(sections):
        raise ConfigError("Section ids must be unique")

    source_packages = profile_cfg.get("source_packages") or []
    if not isinstance(source_packages, list) or not source_packages:
        raise ConfigError("profile.source_packages must be a non-empty list")
    sources = []
    for package in source_packages:
        package_path = root / "config" / "sources" / f"{package}.yaml"
        package_doc = load_yaml(package_path)
        for item in _required_list(package_doc, "sources"):
            source = _parse_source(item, package_path)
            if source.default_section not in section_ids:
                raise ConfigError(
                    f"Source {source.id} uses unknown default_section {source.default_section!r}"
                )
            sources.append(source)
    source_ids = {source.id for source in sources}
    if len(source_ids) != len(sources):
        raise ConfigError("Source ids must be unique across all source packages")

    llm = _parse_llm(_required_mapping(profile_cfg, "llm", f"config/profiles/{profile}.yaml"))
    selection = profile_cfg.get("selection") or {}
    if not isinstance(selection, dict):
        raise ConfigError("profile.selection must be a mapping")

    return AppConfig(
        root=root,
        profile_id=profile_id,
        site=site,
        sections=sections,
        sources=tuple(sources),
        llm=llm,
        selection=selection,
    )


def validate_config(profile: str = "ai-tech", root: Path | None = None) -> AppConfig:
    return load_config(profile=profile, root=root)


def _parse_section(item: Any) -> SectionConfig:
    if not isinstance(item, dict):
        raise ConfigError("Each section must be a mapping")
    section_id = _required_str(item, "id", "section")
    title = _required_str(item, "title", f"section {section_id}")
    keywords = item.get("keywords") or []
    if not isinstance(keywords, list):
        raise ConfigError(f"section {section_id}.keywords must be a list")
    subsections = item.get("subsections") or []
    if not isinstance(subsections, list):
        raise ConfigError(f"section {section_id}.subsections must be a list")
    return SectionConfig(
        id=section_id,
        title=title,
        title_en=str(item.get("title_en") or title),
        icon=str(item.get("icon") or ""),
        max_articles=int(item.get("max_articles") or 6),
        keywords=tuple(str(keyword).lower() for keyword in keywords),
        subsections=tuple(_parse_subsection(subsection, section_id) for subsection in subsections),
    )


def _parse_subsection(item: Any, section_id: str) -> SubsectionConfig:
    if not isinstance(item, dict):
        raise ConfigError(f"Each subsection in section {section_id} must be a mapping")
    subsection_id = _required_str(item, "id", f"subsection in section {section_id}")
    title = _required_str(item, "title", f"subsection {subsection_id}")
    keywords = item.get("keywords") or []
    if not isinstance(keywords, list):
        raise ConfigError(f"subsection {subsection_id}.keywords must be a list")
    return SubsectionConfig(
        id=subsection_id,
        title=title,
        title_en=str(item.get("title_en") or title),
        max_articles=int(item.get("max_articles") or 5),
        keywords=tuple(str(keyword).lower() for keyword in keywords),
    )


def _parse_source(item: Any, path: Path) -> SourceConfig:
    if not isinstance(item, dict):
        raise ConfigError(f"Each source in {path} must be a mapping")
    source_id = _required_str(item, "id", f"source in {path}")
    source_type = _required_str(item, "type", f"source {source_id}").lower()
    if source_type not in {"rss", "atom", "x"}:
        raise ConfigError(f"Source {source_id} has unsupported type {source_type!r}")
    keywords = item.get("keywords") or []
    if not isinstance(keywords, list):
        raise ConfigError(f"source {source_id}.keywords must be a list")
    known = {"id", "name", "type", "enabled", "url", "default_section", "weight", "keywords"}
    return SourceConfig(
        id=source_id,
        name=str(item.get("name") or source_id),
        type=source_type,
        enabled=bool(item.get("enabled", True)),
        url=item.get("url"),
        default_section=_required_str(item, "default_section", f"source {source_id}"),
        weight=float(item.get("weight", 1.0)),
        keywords=tuple(str(keyword).lower() for keyword in keywords),
        options={key: value for key, value in item.items() if key not in known},
    )


def _parse_llm(item: dict[str, Any]) -> LLMConfig:
    return LLMConfig(
        enabled=bool(item.get("enabled", True)),
        provider=str(item.get("provider") or "openai-compatible"),
        api_key_env=str(item.get("api_key_env") or "OPENAI_API_KEY"),
        base_url_env=str(item.get("base_url_env") or "OPENAI_BASE_URL"),
        model_env=str(item.get("model_env") or "OPENAI_MODEL"),
        default_base_url=str(item.get("default_base_url") or "https://api.openai.com/v1"),
        default_model=str(item.get("default_model") or "gpt-4o-mini"),
        temperature=float(item.get("temperature", 0.2)),
        max_input_items=int(item.get("max_input_items", 60)),
        summary_style=str(item.get("summary_style") or "中文简洁摘要"),
    )


def _required_mapping(data: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must define mapping {key!r}")
    return value


def _required_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"Config must define list {key!r}")
    return value


def _required_str(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must define non-empty string {key!r}")
    return value.strip()
