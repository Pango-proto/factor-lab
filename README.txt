factor-lab
==========

A 股量化研究项目：时点数据管道、因子研究、风险建模、回测记账与可视化。
后端使用 Python 3.12；前端使用 React / TypeScript / Vite。

目录职责、上传边界和实验流程详见 docs/project-structure.txt。

快速开始
--------
1. 安装 .python-version 和 .node-version 指定的 Python / Node，以及 uv。
2. 在仓库根目录运行 npm run setup，按 uv.lock 和 package-lock.json 安装依赖。
3. 运行 npm run dev，启动本地开发服务。
4. 配置数据源时参考 .env.example；凭据只保存在本地环境，不提交 Git。

源码验证（不需要下载真实市场数据）
---------------------------------
uv run --locked --extra dev python -m pytest -q -p no:cacheprovider -m "not local_data"
uv run --locked python scripts/check_architecture_invariants.py
uv run --locked python scripts/check_frozen_config_consumption.py
npm run build
node --test tests/chart_geometry.test.mjs

数据与审计边界
--------------
本仓库不上传 data/、reports/、public/demo-backtest/、Markdown 文档、凭据、
依赖目录或缓存。首次克隆不包含真实行情、历史审计证据和演示生成物。
config/ 内已有审计路径及 SHA256 保留原样；这些引用不是随仓库分发的工件。
依赖本地证据的真实研究命令必须恢复匹配的原文件并验签，缺证据时继续阻断。
云端 CI 通过只代表代码与合成测试通过，不代表真实研究准入或 G3 发布门通过。
本地真实证据检查：uv run --locked --extra dev python -m pytest -q -m local_data
本地 G3 发布门：uv run --locked python scripts/check_g3_release_gate.py

可选生成前端合成演示（不是市场实验）
----------------------------------
uv run --locked factor-matrix build-backtest-preview-v3
uv run --locked factor-matrix build-path-ensemble-v1
uv run --locked factor-matrix run-m3-risk-acceptance-v1
后两个命令可能耗时；产物留在 public/demo-backtest/，不提交 Git。
真实数据就绪检查仍要求本地数据，不会因生成演示而开放准入。

分支约定
--------
main：当前项目可审阅的代码基线。
research/fracdiff-v1：分数差分 R0 实验，使用独立工作目录、依赖环境与输出目录。
实验输入必须固定数据版本及工件哈希；Git 分支自身不隔离外部数据。
