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

## 内容质量

正式编纂先选择并翻译文章，再校验 URL、主题证据、重复链接及栏目上限，最后基于最终文章生成小版总结、大版总结和首页总览。

- 栏目没有最低篇数要求，没有合适文章时留空，不跨栏目补位。
- 缺少具体中文标题/摘要、只有模板文案或缺少原文主题证据的条目会被跳过。
- 英文关键词按词边界匹配；来源标签、arXiv 等元数据不作为主题证据。主题判定采用保守筛选，可能减少覆盖率。
- 总结阶段接收最终文章；返回的 URL 集合必须一致。校验失败只重试一次，仍失败则停止生成，保留已有页面。URL 校验不能替代对事实和语义的人工审阅。
- 有文章的大版会增加一次总结请求，空栏目不请求总结。默认八版最多增加八次请求，实际费用取决于入选篇数与模型。
- `--allow-fallback` 本地预览保留原始标题和摘要，不再用通用中文套话冒充翻译。
- 已识别的推理模型使用 `low` 推理强度，并为推理与正文共同预留 16000 token。可通过 `OPENAI_REASONING_EFFORT`、`OPENAI_MAX_TOKENS` 覆盖；其他模型保留原有默认。参数依据：[OpenAI 推理指南](https://developers.openai.com/api/docs/guides/reasoning)、[Gemini 兼容接口](https://ai.google.dev/gemini-api/docs/openai)。
- API 返回截断、非法 JSON 或总结校验失败时，同一阶段最多尝试两次，失败时不渲染新页面。

### 离线编纂审阅版

没有本地 API 凭据时，可以审阅基于已保存抓取快照、由 Codex 编纂的版本：

```bash
PYTHONPATH=src python3 scripts/render_review.py \
  --snapshot preview/2026-10-12/snapshot.json \
  --editorial preview/2026-10-12/editorial.json \
  --out preview/2026-10-12/site
```

审阅版使用与正式流程相同的文章及总结校验，输出保存在忽略的 `preview/` 中，不改写正式日报数据。`editorial.json` 包含具体选稿、原文主题证据和基于最终文章重新编写的总结；`site/edition.json` 保存最终内容及编纂来源。离线审阅版不代表在线模型 API 已完成端到端验证。
