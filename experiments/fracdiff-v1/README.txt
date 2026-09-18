分数差分 R0 独立实验工作区
=========================

当前分支：research/fracdiff-v1
状态：分支及工作目录已建立；数据刷新未完成，实验尚未运行。
机器可读边界：experiments/fracdiff-v1/workspace.json。

main 保留项目基线。本分支添加实现、研究依赖锁、测试和执行接口。
请从此 worktree 根目录执行命令，避免回到原项目目录修改代码或写实验结果。
依赖可用 uv sync --locked --extra dev 安装到自己的 .venv。
statsmodels 等研究依赖仍需按方案单独锁定；本次没有随意添加版本。

配置 config/fracdiff_research_proposal_v1.json 保持原始参数及哈希。
其 draft 状态是最初方案记录；本次执行授权单独记录在 workspace.json。
原始 Markdown 和审阅包按用户要求只保存在本地，未改名绕过上传限制。
在其他电脑克隆时，应单独恢复这些证据并核对方案/批准/输入哈希。

数据更新与实验顺序
------------------
1. 在原项目完成已接入数据更新和质量验收。
2. 显式绑定新的数据 manifest、版本、实际工件 SHA256；不继续引用旧 latest。
3. 通过只读快照或独立副本消费输入；本工作区不创建指向原 data/ 的可写软链接。
4. 独立实现 diagnose-fracdiff-v1 服务入口；当前命令尚不存在。
5. 按方案运行合成验收、输入支持集审计、训练筛选和固定后段验证。
6. 结果写入本工作区 data/diagnostics/fracdiff_v1 和 reports/fracdiff_v1。

本地输入路径另写 runtime.local.json，不提交 Git。
Git 和 worktree 隔离代码；它们不会自动提供数据只读权限或历史数据快照。
日期上限 2025-02-28，Holdout 继续封存；本轮仅 R0，不执行 Alpha 或策略回测。
