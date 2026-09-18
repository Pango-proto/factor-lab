#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/context_maintenance.sh                                   # only regenerate index/report
#   ./scripts/context_maintenance.sh --clean                             # clean cache artifacts
#   ./scripts/context_maintenance.sh --retain-gold-runs N                 # retain latest N gold run_id dirs
#   ./scripts/context_maintenance.sh --retain-diagnostics-days D           # retain diagnostics in last D days
#   ./scripts/context_maintenance.sh [--clean] [--retain-gold-runs N] [--retain-diagnostics-days D] [--archive-root DIR]
#   ./scripts/context_maintenance.sh --help                               # show usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_FILE="$ROOT_DIR/PROJECT_CONTEXT_INDEX.md"
DATA_POLICY_FILE="$ROOT_DIR/data/metadata/data_layer_policy.md"
DO_CLEAN=0
RETAIN_GOLD_RUNS=0
RETAIN_DIAGNOSTICS_DAYS=0
ARCHIVE_ROOT="$ROOT_DIR/data/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --retain-gold-runs)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "Missing or invalid value for --retain-gold-runs, expected non-negative integer." >&2
        exit 1
      fi
      RETAIN_GOLD_RUNS="$2"
      shift 2
      ;;
    --retain-diagnostics-days)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "Missing or invalid value for --retain-diagnostics-days, expected non-negative integer." >&2
        exit 1
      fi
      RETAIN_DIAGNOSTICS_DAYS="$2"
      shift 2
      ;;
    --archive-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --archive-root." >&2
        exit 1
      fi
      ARCHIVE_ROOT="$2"
      shift 2
      ;;
    --help)
      echo "Usage:"
      echo "  ./scripts/context_maintenance.sh                                   # regenerate context index"
      echo "  ./scripts/context_maintenance.sh --clean                             # regenerate index + clean cache noise"
      echo "  ./scripts/context_maintenance.sh --retain-gold-runs N                 # retain latest N gold run_id dirs"
      echo "  ./scripts/context_maintenance.sh --retain-diagnostics-days D           # retain diagnostics in last D days"
      echo "  ./scripts/context_maintenance.sh [--clean] [--retain-gold-runs N] [--retain-diagnostics-days D] [--archive-root DIR]"
      echo "  ./scripts/context_maintenance.sh --help                               # show usage"
      echo
      echo "Examples:"
      echo "  ./scripts/context_maintenance.sh --retain-gold-runs 4 --retain-diagnostics-days 14"
      echo "  ./scripts/context_maintenance.sh --clean --retain-gold-runs 5"
      echo "  ./scripts/context_maintenance.sh --retain-gold-runs 6 --archive-root data/archive"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Run: ./scripts/context_maintenance.sh --help" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"
export LC_ALL=C

generate_data_policy() {
  mkdir -p "$(dirname "$DATA_POLICY_FILE")"
  {
    echo "# 数据分层与使用策略建议"
    echo
    echo "更新时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo
    echo "## 不建议改动（Immutable，拉取后只读）"
    echo "- data/bronze/**"
    echo "- data/metadata 的静态文件与基线快照（如 catalog/registry/snapshot）"
    echo
    echo "## 回测可复用（建议保留版本）"
    echo "- data/gold/l2b_risk_only/**"
    echo "- data/gold/risk_exposure_matrix/**"
    echo "- data/gold/tradable_universe/**"
    echo "- data/gold/board_benchmarks/**"
    echo
    echo "## 低优先可清理/归档（不建议长期占上下文）"
    echo "- data/diagnostics/**（保留近期即可）"
    echo "- 重复/过老的 run_id 产物"
    echo
    echo "执行建议:"
    echo "- 回测相关：按策略保留最近 2~4 个版本"
    echo "- 归档相关：将其余 run_id move 到离线归档，不放工作区"
    echo
    echo "当前脚本策略参数:"
    echo "- RETAIN_GOLD_RUNS=${RETAIN_GOLD_RUNS:-0}"
    echo "- RETAIN_DIAGNOSTICS_DAYS=${RETAIN_DIAGNOSTICS_DAYS:-0}"
    echo "- ARCHIVE_ROOT=$ARCHIVE_ROOT"
  } > "$DATA_POLICY_FILE"
}

collect_run_dirs() {
  local base_dir="$1"
  find "$base_dir" -mindepth 1 -maxdepth 1 -type d -name 'run_id=*' -print
}

archive_run_dirs_if_needed() {
  local base_dir="$1"
  local retain="$2"
  if (( retain <= 0 )); then
    return 0
  fi
  if [ ! -d "$base_dir" ]; then
    return 0
  fi

  local rel_dir="${base_dir#"$ROOT_DIR/"}"
  local archive_dir="$ARCHIVE_ROOT/$rel_dir"
  mkdir -p "$archive_dir"

  mapfile -t run_dirs < <(collect_run_dirs "$base_dir" | sort)
  local total=${#run_dirs[@]}
  if (( total <= retain )); then
    return 0
  fi

  local to_archive=$(( total - retain ))
  local sorted_by_mtime
  mapfile -t sorted_by_mtime < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -name 'run_id=*' -print0 | xargs -0 ls -1dt 2>/dev/null | sed 's|/$||')
  local i=0
  for dir in "${sorted_by_mtime[@]}"; do
    if (( i >= retain )); then
      local target="$archive_dir/$(basename "$dir")"
      local stamp="$(date -u '+%Y%m%d')"
      # avoid overwrite when duplicate names
      while [ -e "$target" ]; do
        target="${archive_dir}/${stamp}_$(basename "$dir")"
      done
      mv "$dir" "$target"
    fi
    ((i++))
  done
}

archive_old_diagnostics_if_needed() {
  local diagnostics_dir="$1"
  local keep_days="$2"
  if (( keep_days <= 0 )); then
    return 0
  fi
  if [ ! -d "$diagnostics_dir" ]; then
    return 0
  fi

  local keep_cutoff
  keep_cutoff="$(date -u -d "$keep_days days ago" +%s 2>/dev/null || date -v -${keep_days}d -u +%s)"
  local archive_dir="$ARCHIVE_ROOT/diagnostics"
  mkdir -p "$archive_dir"

  while IFS= read -r -d '' entry; do
    if [ -f "$entry" ] || [ -d "$entry" ]; then
      local ts
      ts="$(stat -c %Y "$entry" 2>/dev/null || stat -f %m "$entry")"
      if (( ts < keep_cutoff )); then
        local rel="${entry#"$ROOT_DIR/data/diagnostics/"}"
        local target_dir="$archive_dir/$rel"
        mkdir -p "$(dirname "$target_dir")"
        mv "$entry" "$target_dir"
      fi
    fi
  done < <(find "$diagnostics_dir" -mindepth 2 -maxdepth 2 -type d -name 'run_id=*' -print0)
}

{
  echo "# 项目上下文整理索引"
  echo
  echo "生成时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo
  echo "## 1) 开发核心（建议优先阅读）"
  echo "目录: frontend, backend, scripts, config, tests"
  echo
  {
    rg --files frontend backend scripts config tests
    find . -maxdepth 1 -type f -name "*.md" -print
  } | sed 's|^\./||' | sort -u
  echo

  echo "## 2) 体量过滤（排除噪声目录后）"
  echo "命令: rg --files | rg -v '^(node_modules/|\\.venv/|data/|dist/|\\.pytest_cache/|.*/__pycache__/|.*\\.pyc$|.*/\\.DS_Store$|^\\.DS_Store$)'"
  echo
  rg --files | rg -v '^(node_modules/|\\.venv/|data/|dist/|\\.pytest_cache/|.*/__pycache__/|.*\\.pyc$|.*/\\.DS_Store$|^\\.DS_Store$)' | sort | head -n 200
  echo

  echo "## 3) 最大体积文件（排除主要噪声目录后）"
  echo "命令: 文件体积降序（字节）"
  echo
  rg --files | rg -v '^(node_modules/|\\.venv/|dist/|\\.pytest_cache/|data/)' | while IFS= read -r f; do
    if [ -f "$f" ]; then
      printf "%10s\t%s\n" "$(wc -c < "$f")" "$f"
    fi
  done | sort -nr | head -n 120
  echo

  echo "## 4) Data 分层治理（Bronze / Silver / Gold）"
  echo
  if [ -d data ]; then
    echo "### data 规模"
    du -sh data
    echo
    for layer in bronze silver gold diagnostics metadata; do
      if [ -d "data/$layer" ]; then
        echo "### data/$layer"
        du -sh "data/$layer"
        echo
      fi
    done
    echo "### Golden 层回测优先目录（示例，按 run_id 归档）"
    for path in \
      "data/gold/l2b_risk_only" \
      "data/gold/risk_exposure_matrix" \
      "data/gold/tradable_universe" \
      "data/gold/board_benchmarks"; do
      if [ -d "$path" ]; then
        echo "#### $path"
        find "$path" -type d -name 'run_id=*' | sort | head -n 20
      fi
    done
    echo
  else
    echo "data 目录不存在"
    echo
  fi
} > "$INDEX_FILE"

generate_data_policy

echo "已更新: $INDEX_FILE"
echo "已更新: $DATA_POLICY_FILE"

  if [[ "$DO_CLEAN" == "1" ]]; then
  echo "开始清理上下文噪声文件..."
  # Python/测试缓存（可安全重建）
  for d in backend scripts tests; do
    if [ -d "$d" ]; then
      find "$d" -type d -name "__pycache__" -prune -exec rm -rf {} +
      find "$d" -type f -name "*.pyc" -delete
    fi
  done

  find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
  find . -type f -name "*.pyc" -delete
  find . -type f -name ".DS_Store" -delete
  rm -f tsconfig.tsbuildinfo
  # dist 可通过 npm build 复建，减少读盘干扰
  rm -rf dist
  echo "清理完成。"
else
  echo "仅生成索引文件，未执行清理。"
fi

if (( RETAIN_GOLD_RUNS > 0 )); then
  echo "开始执行 gold run_id 保留策略（保留最近 ${RETAIN_GOLD_RUNS} 个）..."
  for path in \
    "data/gold/l2b_risk_only" \
    "data/gold/risk_exposure_matrix" \
    "data/gold/tradable_universe" \
    "data/gold/board_benchmarks"; do
    if [ -d "$path" ]; then
      archive_run_dirs_if_needed "$path" "$RETAIN_GOLD_RUNS"
    fi
  done
fi

if (( RETAIN_DIAGNOSTICS_DAYS > 0 )); then
  echo "开始执行 diagnostics 保留策略（保留最近 ${RETAIN_DIAGNOSTICS_DAYS} 天）..."
  archive_old_diagnostics_if_needed "data/diagnostics" "$RETAIN_DIAGNOSTICS_DAYS"
fi
