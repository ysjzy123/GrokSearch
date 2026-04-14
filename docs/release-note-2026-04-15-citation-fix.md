# GrokSearch 更新说明（2026-04-15）

## 变更概述

本次更新围绕“在尽量控制延迟和预算的前提下，提升聚合搜索质量、复杂查询综合能力，以及 citation 支撑一致性”展开。

核心目标：

- 默认模型统一到 `grok-4.20-fast`
- 删除 `GUDA_*` 相关空配置和回退逻辑，配置入口收敛为标准 `GROK/TAVILY/FIRECRAWL` 变量
- 为复杂查询引入分阶段模型策略：搜索阶段偏快，综合阶段偏稳
- 优化 multifacet / learning / roadmap 类查询的外部聚合、排序、综合与兜底
- 修复答案内联引用没有回写到缓存 sources 的问题，消除 citation precision 被低估的问题

## 主要改动

### 1. 默认模型调整

- 默认模型改为 `grok-4.20-fast`
- 搜索阶段默认保持 fast
- 对复杂综合阶段按条件切换到同系 `auto` 模型

这样做的目的不是全链路升模，而是只在真正需要综合判断时提高稳定性，避免简单查询无谓变慢。

### 2. 配置收敛

移除了 `GUDA_BASE_URL` / `GUDA_API_KEY` 派生式配置路径，避免：

- 空配置项长期保留
- 不同配置来源相互覆盖
- 诊断时难以确认实际生效参数

当前以显式配置为准：

- `GROK_API_URL`
- `GROK_API_KEY`
- `TAVILY_API_URL`
- `TAVILY_API_KEY`
- `FIRECRAWL_API_URL`
- `FIRECRAWL_API_KEY`

### 3. 复杂查询的 staged model 机制

新增分阶段模型策略：

- `search_model = grok-4.20-fast`
- `analysis_model = grok-4.20-auto`

触发条件主要包括：

- multifacet 查询
- learning / roadmap 查询
- planned queries 大于 1
- 明确偏 source synthesis 的路径

这样可以在不显著拉高简单查询成本的前提下，提高复杂查询的综合质量。

### 4. 聚合与综合流程增强

对复杂查询新增或强化了以下能力：

- 外部 sources 扩展预算拆分
- follow-up query 扩展
- source enrichment
- source ranking
- source synthesis
- fallback Grok search
- 更完整的 trace / postprocessing 信息

同时保留了失败时的 sources-only fallback，避免返回空内容。

### 5. citation support 一致性修复

修复前：

- 回答正文可能包含新的内联 citations
- 但这些 URL 没有并回 `all_sources`
- 导致 `citation_source_support_ratio` 和 `citation_precision` 被低估

修复后：

- 在返回前统一提取答案中的内联 citation URL
- 与已有 `all_sources` 执行 merge
- trace 中增加 `inline_citation_sources_count`

这保证了：

- sources cache 与最终答案一致
- 评测层能正确识别 citation support
- 下游 `get_sources`、evidence 绑定和评分逻辑更一致

## 正式 A/B 结果

对比对象固定为：

- baseline：`GuDaStudio/GrokSearch@grok-with-tavily`
- current：本地最新修改版

### non-live 正式 A/B

- gate: `keep`
- current: `127.66`
- baseline: `112.20`

结论：

- 本地版总分领先 `15.46`
- citation precision 非回退
- 预算明显低于原版

### include-live 正式 A/B

- gate: `keep`
- current: `141.84`
- baseline: `126.84`

结论：

- 本地版总分领先 `15.00`
- retrieval robustness 持平
- live efficiency 更优

## 综合判断

当前版本相对于 `GuDaStudio/grok-with-tavily` 的实际状态：

- 搜索质量更高
- 复杂查询综合更稳
- citation 支撑更完整
- 预算使用更低
- 正式评分已通过 gate

## 对使用者的影响

你将观察到的主要变化：

- 默认模型为 `grok-4.20-fast`
- 复杂查询更容易进入“fast 搜索 + auto 综合”的模式
- 多来源综合答案的 citations 更一致
- 聚合失败时更少出现空内容
- trace 信息更完整，更利于诊断
