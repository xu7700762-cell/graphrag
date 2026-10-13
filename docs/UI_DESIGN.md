# EEG 科研工作台 UI

2026-10-09 实施，原生 HTML / CSS / JavaScript，无新增前端运行依赖。

## 设计方向

浅色科研仪器风。用分隔线、清晰的文字层级和证据颜色组织内容；协作对话是主工作区，实验参数与评价结果有独立位置。正文左对齐，避免以装饰性标语替代真实状态。

| 角色 | 色值 |
| --- | --- |
| 页面背景 | `#F3F6FA` |
| 面板 | `#FFFFFF` |
| 正文 | `#182A44` |
| 次级文字 | `#52647B` |
| 操作 | `#175CD3` |
| 证据 | `#007C83` |

字体为微软雅黑 UI，回退到微软雅黑 / Segoe UI。正文 16px，控件 14px，辅助文字不小于 12px；数值使用 tabular-nums。控件提供可见键盘焦点，主要按钮最小高度 44px，尊重 reduced-motion。

## 页面及代码

| 文件 | 职责 |
| --- | --- |
| `eeg_agent/web/design-tokens.css` | 共用色彩、字体、圆角与焦点变量；不包含全局元素规则 |
| `brain.html` / `brain.css` | 工作台四个页面及响应式布局 |
| `brain.js` | 对话、证据、目录、评价渲染及工作区地址状态 |
| `index.html` / `app.js` | 保留 EEG 监测入口，提供双模型配置及保存状态 |
| `cloud.css` | 仅对 `body[data-page="cloud"]` 生效；监测控制台不采用此页样式 |

工作区地址为 `/#conversation`、`/#evaluation`、`/#toolkits`、`/#knowledge`，API 入口为 `/monitor#cloud`。切换页面使用 History API，支持浏览器返回、刷新及直接进入。启动时后台恢复历史对话，不覆盖用户请求的页面。

对话页的 `questionForm` 只包含输入与发送。研究路径、领域范围、语言后端及辅助测量收纳于研究参数折叠区。评价页保留原 `vrmsSeed`；对话中的 `researchSeed` 同步该值和运行锁，避免两页使用不同种子。

评价区保留全局活动批次的恢复、轮询、完整 146 路径分母与原续跑条件。只有 completed 状态显示最终 ACC。历史研究对照与网页评价记录分开渲染，显示实际记录、模型及续跑来源。

小于 1280px 时，协作证据由按钮打开；小于 1100px 时，侧栏为抽屉并提供四页快捷导航。窄屏保持对话区最小可读高度，允许整页滚动。参数展开不压缩对话到不可读的高度。

## 设计技能版本

经官方 skill-installer 安装并读取以下固定版本。本项目采用用户确认的配色和研究工作区结构；技能的营销页建议未直接套用。

- [frontend-design](https://github.com/anthropics/skills/tree/683bc88e56f3e09ba94f7055977f3d3aa499f202/skills/frontend-design)，提交 `683bc88e56f3e09ba94f7055977f3d3aa499f202`。
- [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/tree/1a2c459b35f26116fd165b0a0f30597f252749ff)，提交 `1a2c459b35f26116fd165b0a0f30597f252749ff`。

## 回归与边界

`node tests/test_brain_ui.js` 验证导航、恢复、指标分母、运行锁、续跑条件、空状态及 Markdown 下载内容。`node tests/test_cloud_settings_ui.js` 验证双档案、独立密钥、保存与失败状态。相关后端测试采用本地替身，不能据此宣称新的模型效果。

本次不改变分类模型、训练、数据协议、检索实现或 API 合同；不启动训练和付费评价。完整验收及截图见 `outputs/brain_workspace/ui_redesign_20261009/`，原六个网页文件备份在其 `source_backup/` 下。
