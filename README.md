# factor-lab

一个面向 A 股的量化研究项目，涵盖时点数据处理、因子研究、风险建模、回测记账和可视化。后端使用 Python 3.12，前端使用 React、TypeScript 和 Vite。

关于目录结构、版本控制范围和实验流程，请参阅[项目结构说明](docs/project-structure.txt)。

## 项目结构

| 目录 | 用途 |
| --- | --- |
| `frontend/` | React / TypeScript 前端，负责页面、图表和交互。 |
| `backend/factor_matrix/` | Python 后端，负责数据处理、因子计算、风险建模、回测和 API。 |
| `config/` | 模型参数、研究协议和审计绑定配置。 |
| `scripts/` | 环境初始化、开发启动和检查脚本。 |
| `tests/` | 前后端测试。 |
| `docs/` | 项目结构和协作说明。 |
| `public/` | 前端静态资源。 |

前后端共用仓库根目录的启动入口。`package.json`、`pyproject.toml` 和依赖锁文件也保存在根目录，以下命令均在这里运行。

## 当前进展（2026-09-18）

风险准入已拆分为基底、协方差和真实时点证据三个独立状态，支持证据校验、登记库增量迁移，以及通过只读接口 `/api/risk-acceptance` 查询。Silver 读取器支持显式固定版本，独立审计不会随最新数据版本切换输入。

六风格模型已完成独立权重与正交性复算、残差归因和缺失收益事件核对。数值复算通过，但正式基底验收仍有角色迁移和历史研究成本证据缺口；协方差校准未通过，真实时点证据也未补齐。当前结果不构成生产或风险优化许可。

分数差分 R0 已在 `research/fracdiff-v1` 的独立工作区完成本地实验与结果核验，结果说明位于该分支的 `experiments/fracdiff-v1/results/20260918/README.md`。实验实现尚未合并到 `main`，也未进入 Alpha 注册或策略回测。

## 快速开始

1. 按照 `.python-version` 和 `.node-version` 安装对应版本的 Python 和 Node.js，并安装 `uv`。
2. 在仓库根目录安装依赖。此命令会使用 `uv.lock` 和 `package-lock.json` 中锁定的版本：

   ```bash
   npm run setup
   ```

3. 启动本地开发服务：

   ```bash
   npm run dev
   ```

4. 如需配置数据源，请参考 [`.env.example`](.env.example)。凭据应保存在本地环境中，不要提交到 Git。

## 验证代码

以下检查不需要下载真实市场数据，可用于验证源码、合成测试和前端构建：

```bash
uv run --locked --extra dev python -m pytest -q -p no:cacheprovider -m "not local_data"
uv run --locked python scripts/check_architecture_invariants.py
uv run --locked python scripts/check_frozen_config_consumption.py
npm run build
node --test tests/chart_geometry.test.mjs
```

## 数据与研究验证

仓库不包含真实行情、历史审计证据或生成的演示数据。`data/`、`reports/`、`public/demo-backtest/`、本地 Markdown 文档（根目录的 `README.md` 除外）、凭据、依赖目录和缓存均不纳入版本控制。

`config/` 中保留了原有的审计路径和 SHA256 哈希，但这些路径引用的文件不随仓库分发。运行依赖本地证据的真实研究命令前，需要恢复对应的原始文件并通过哈希校验；如果证据缺失，检查仍会阻止后续运行。

源码目录已统一命名为 `frontend/` 和 `backend/`，配置中的源码引用也已同步。部分研究标识会将源码路径或配置内容纳入哈希，因此目录迁移后的新运行可能生成不同的标识。历史数据、报告及其审计记录保持原样，复现历史运行时应使用对应的代码版本。

GitHub CI 通过表示代码检查和合成测试通过，不代表真实研究已满足准入要求，也不代表 G3 发布门检查通过。基底、协方差和真实时点证据分别验收；旧登记记录中的 `frozen` 不会自动转为新认证通过。

在具备完整本地数据和证据的环境中，可运行以下检查：

```bash
# 检查真实本地证据
uv run --locked --extra dev python -m pytest -q -m local_data

# 检查 G3 发布门
uv run --locked python scripts/check_g3_release_gate.py
```

## 生成前端演示（可选）

以下命令用于生成前端展示所需的合成数据，不属于真实市场实验：

```bash
uv run --locked factor-matrix build-backtest-preview-v3
uv run --locked factor-matrix build-path-ensemble-v1
uv run --locked factor-matrix run-m3-risk-acceptance-v1
```

后两个命令可能需要较长时间。生成文件保存在 `public/demo-backtest/`，不提交到 Git。演示生成完成后，真实数据的就绪检查仍需单独通过。

## 分支约定

- `main`：当前项目可供审阅的代码基线。
- `research/fracdiff-v1`：分数差分 R0 实验，使用独立的工作目录、依赖环境和输出目录，运行方式以该分支的实验说明为准。

`frontend/` 和 `backend/` 是 `main` 当前的目录结构。实验分支仍保留自己的源码布局；后续合并时需要同步适配路径和依赖配置。

实验输入需要固定数据版本和工件哈希。Git 分支只隔离代码，外部数据仍需通过独立目录、只读快照或独立副本进行隔离。
