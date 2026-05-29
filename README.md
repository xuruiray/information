# 信息日报

一个可配置方向的个人日报生成器：从 RSS/Atom 等数据源抓取信息，按 profile 筛选、去重、编纂为中文日报，并发布为 GitHub Pages 复古报纸页面。

## 本地使用

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

python -m information_daily validate-config --profile ai-tech
python -m information_daily generate --profile ai-tech --out docs --allow-fallback
```

生产生成默认需要 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
python -m information_daily generate --profile ai-tech --out docs
```

`--allow-fallback` 只用于本地检查版式。正式日报必须配置 LLM，因为新闻标题和摘要需要翻译成中文。

## 配置

- `config/site.yaml`：站点标题、时区、归档设置、GitHub Pages 信息。
- `config/profiles/ai-tech.yaml`：默认新闻方向。后续可以复制成财经、国际、产品等新方向。
- `config/sources/*.yaml`：可复用数据源包。RSS/Atom 已实现，`x` 类型为预留接口。

## 自动发布

`.github/workflows/daily.yml` 会在北京时间每天 08:00 运行，也支持手动触发并填写 profile。
