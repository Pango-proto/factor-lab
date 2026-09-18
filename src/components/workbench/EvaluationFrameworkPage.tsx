import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CircleDashed,
  FlaskConical,
  LockKeyhole,
  ShieldCheck,
} from 'lucide-react'
import type { EvaluationFramework } from '../../types'
import { WorkspaceHeader } from '../WorkspaceHeader'

const labels: Record<string, string> = {
  pinned_v1_artifact_lineage: '绑定原评估与独立诊断来源',
  independent_OLS_and_rank_replay_audit: '独立中性化与秩相关复算',
  structured_permutation_diagnostics: '置换支持集与结构贡献诊断',
  separate_execution_implementation_temporal_research_states: '工程、时序与研究状态分别记录',
  fail_closed_temporal_validation: '时序尚未校准时保持阻断',
  preregister_temporal_null_calibration: '预登记时序零假设校准方案',
  validate_single_candidate_on_synthetic_null_and_power_cases: '合成样本检验误拒绝率与检测能力',
  review_evidence_before_real_data_inference: '真实样本推断前复核校准证据',
  ic_mean_median_sd_ir: 'IC 均值 / 中位数 / 波动 / IR',
  newey_west_standard_error_and_t: 'Newey–West 标准误与 t 值',
  daily_ic_and_rolling_12m: '日频 IC 与滚动 12 月均值',
  ic_decay_curve: '1 / 3 / 5 / 10 / 20 日衰减曲线',
  negative_control_gate: '按当前契约登记的负对照闸门',
  label_and_raw_signal_alignment_shift_curves: '标签 / 原始信号对齐平移曲线',
  time_shuffle_100_repetition_null_distribution: '100 次时序置换 null 分布',
  daily_effective_sample_distribution: '每日有效样本分布',
  time_shuffle_autocorrelation_diagnostics: '时序打乱自相关诊断',
  holdout_and_purge_audit: 'Holdout 与 purge 边界审计',
  liquidity_and_size_quantile_breakdown: '流动性 / 市值分位拆解',
  next_day_execution_filter: '下一交易日可执行性过滤',
  signal_vs_neutralization_turnover: '信号 / 中性化换手拆解',
  decile_portfolio_nav: '十分位组合净值',
  segmented_cost_model: '分段交易成本模型',
  walk_forward_segments: 'Walk-forward 分段检验',
}

const percentage = (value: number | undefined, digits = 2) => value !== undefined && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
const decimal = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—'

export function EvaluationFrameworkPage() {
  const [framework, setFramework] = useState<EvaluationFramework | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('/api/evaluation/framework')
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json()
      })
      .then(setFramework)
      .catch((reason) => setError(String(reason)))
  }, [])

  const primary = framework?.contract.sample.primary_horizon ?? 5
  const candidate = framework?.latest_run
  const run = framework?.status === 'ready'
    && candidate && ['completed', 'passed'].includes(candidate.status)
    && candidate.negative_controls.passed ? candidate : null
  const excluded = framework?.excluded_run
  const blocked = framework?.status === 'blocked'
  const waitingLabel = blocked ? '研究检验受阻 · 主指标暂不可用' : excluded ? '暂无有效结果 · 等待重新评估' : '契约已登记 · 等待首跑'
  const excludedReason = excluded?.reason
    .replaceAll('date_axis_shuffle', '日期轴置换')
    .replaceAll('within_asset_time_shuffle_demeaned', '去均值时序置换')
    .replaceAll('within_date_cross_sectional_return_shuffle', '截面收益打乱')
    .replaceAll('label_window_shift', '标签窗口平移')
  const labelShift = run?.negative_controls.label_window_shift_peak_at_zero
  const rawShift = run?.negative_controls.raw_signal_shift_peak_at_zero
  const timeShuffle = run?.negative_controls.within_asset_time_shuffle_100_repetition_asset_demeaned_null
  const rawShiftIsGate = framework?.contract.evaluation_channel.variant_dispositions?.raw_signal_shift === 'gate'
  const primaryIc = run?.ic.by_horizon[String(primary)]
  const curveMax = useMemo(
    () => Math.max(0.01, ...(run?.ic.decay_curve.map((point) => Math.abs(point.mean)) ?? [])),
    [run],
  )

  return (
    <main className="workbench-main evaluation-page">
      <WorkspaceHeader
        eyebrow="P0 · EVALUATION FRAMEWORK"
        title="估计区间评估框架"
        description="先验证信号读数可信，再进入成本、频率与风险模型比较。"
        aside={(
          <div className={`evaluation-gate ${run?.negative_controls.passed ? 'passed' : 'waiting'}`}>
            {run?.negative_controls.passed ? <ShieldCheck size={18} /> : <CircleDashed size={18} />}
            <div>
              <strong>{run ? '负对照通过' : waitingLabel}</strong>
              <span>Holdout 保持封存</span>
            </div>
          </div>
        )}
      />

      {error ? <div className="workbench-error">评估框架读取失败：{error}</div> : null}
      {framework?.review ? (
        <section className="evaluation-metrics" aria-label="研究验收状态">
          <article><span>工程审计</span><strong>{framework.review.implementation_status === 'passed' ? '通过' : '未通过'}</strong><small>独立复算与来源核验</small></article>
          <article><span>结构诊断</span><strong>已完成</strong><small>两项原失败记录保留</small></article>
          <article><span>时序有效性</span><strong>尚未校准</strong><small>等待预登记校准验证</small></article>
          <article><span>研究验收</span><strong>仍受阻</strong><small>暂不可推进评分与优化</small></article>
        </section>
      ) : null}
      {excluded ? (
        <div className="workbench-error evaluation-status" role="status">
          <strong>{excludedReason}</strong>
          {excluded.superseded_by ? <p>替代版本：{excluded.superseded_by}</p> : null}
          <p>{blocked ? '诊断记录已保留，主指标暂停展示。' : '旧结果保留在历史记录中，不参与当前指标展示。'}</p>
        </div>
      ) : null}
      {!framework ? <div className="evaluation-loading"><CircleDashed size={18} />读取评估契约…</div> : (
        <>
          <section className="evaluation-boundary">
            <div className="boundary-copy">
              <span><LockKeyhole size={14} />样本边界</span>
              <strong>{framework.contract.sample.development_start} — {run?.scope.end ?? (blocked ? '运行边界见验收记录' : '安全终点待首跑')}</strong>
              <p>最大预测期 {Math.max(...framework.contract.sample.reported_horizons)} 个交易日 + embargo {framework.contract.sample.embargo_trading_days} 日，自动导出 purge gap {framework.contract.sample.purge_gap_trading_days} 日。</p>
            </div>
            <ArrowRight size={18} />
            <div className="holdout-seal">
              <span>SEALED HOLDOUT</span>
              <strong>{framework.contract.sample.holdout_start} → 今天</strong>
              <small>{framework.contract.holdout_capacity.as_of ? `截至 ${framework.contract.holdout_capacity.as_of} ` : ''}约 {framework.contract.holdout_capacity.approximate_trading_days} 个交易日 · 仅供最终灾难性失败检查</small>
            </div>
          </section>

          <section className="evaluation-metrics">
            <article>
              <span>{primary} 日平均 IC</span>
              <strong>{primaryIc ? decimal(primaryIc.mean, 4) : '—'}</strong>
              <small>{primaryIc ? `${primaryIc.observations.toLocaleString()} 个截面` : blocked ? '检验未通过，暂不展示' : '等待正式首跑'}</small>
            </article>
            <article>
              <span>IC_IR</span>
              <strong>{primaryIc ? decimal(primaryIc.ic_ir) : '—'}</strong>
              <small>mean / sd · 不用重叠窗口虚高 t</small>
            </article>
            <article>
              <span>Newey–West t</span>
              <strong>{primaryIc ? decimal(primaryIc.newey_west_t, 2) : '—'}</strong>
              <small>lag = {primaryIc?.newey_west_lag ?? primary} ≥ horizon</small>
            </article>
            <article>
              <span>日均换手</span>
              <strong>{percentage(run?.backtest?.mean_turnover, 1)}</strong>
              <small>{run?.backtest ? '粗略 top-decile 组合' : '等待有效组合评估结果'}</small>
            </article>
          </section>

          <div className="evaluation-grid">
            <section className="evaluation-card controls-card">
              <div className="evaluation-card-heading">
                <div><span>T1 · REQUIRED GATES</span><h2>负对照</h2></div>
                <b className={run?.negative_controls.passed ? 'ok' : 'warn'}>{run ? '全部通过' : blocked ? '存在阻断' : '等待首跑'}</b>
              </div>
              {framework.review ? <SimpleGateRow title="截面内收益打乱" note="原五期限必需门结果"
                passed={framework.review.required_gates.within_date_cross_sectional_return_shuffle} /> : <ControlRow
                title="截面内收益打乱"
                note="保留日期结构，摧毁当日资产对应关系"
                value={run?.negative_controls.within_date_cross_sectional_return_shuffle?.[String(primary)]}
              />}
              {framework.review ? <SimpleGateRow title="日期轴置换 · 结构诊断"
                note="支持集变化与持久排序已核对；原零中心门未通过" detail="原失败保留" /> : null}
              {framework.contract.evaluation_channel.variant_dispositions?.date_axis_shuffle === 'gate' ? (
                <ControlRow
                  title="日期轴打乱"
                  note="保留完整截面，检查日期对应关系"
                  value={run?.negative_controls.whole_cross_section_time_axis_shuffle?.[String(primary)]}
                />
              ) : null}
              <SimpleGateRow
                title="标签窗口平移"
                note="按当前契约导出的偏移网格检查标签窗口对齐"
                passed={framework.review?.required_gates.label_window_shift ?? labelShift?.passed}
                detail={labelShift ? `峰值 ${labelShift.peak_offsets_trading_days.join(', ')}` : undefined}
              />
              <SimpleGateRow
                title={rawShiftIsGate ? '原始信号平移' : '原始信号平移 · 敏感性观察'}
                note={rawShiftIsGate ? '按当前契约检查信号平移门' : '当前契约不将此项作为独立通过门槛'}
                passed={rawShiftIsGate ? rawShift?.passed : undefined}
                detail={rawShift ? `峰值 ${rawShift.peak_offsets_trading_days.join(', ')}` : undefined}
              />
              <SimpleGateRow
                title={framework.review ? '去均值时序置换 · 结构诊断' : '100 次去均值时序置换'}
                note={framework.review ? '去收益均值不保证秩均值为零；原零中心门未通过' : '逐股票去长期标签均值后独立打乱；95% null 区间须覆盖 0'}
                passed={timeShuffle?.passed}
                detail={framework.review ? '原失败保留' : timeShuffle ? `${decimal(timeShuffle.lower, 4)} — ${decimal(timeShuffle.upper, 4)}` : undefined}
              />
            </section>

            <section className="evaluation-card decay-card">
              <div className="evaluation-card-heading">
                <div><span>T2 · TERM STRUCTURE</span><h2>IC 衰减曲线</h2></div>
                <small>交易日</small>
              </div>
              {run?.ic.decay_curve.length ? run.ic.decay_curve.map((point) => (
                <div className="decay-row" key={point.horizon_trading_days}>
                  <b>{point.horizon_trading_days}D</b>
                  <div><i style={{ width: `${Math.max(3, Math.abs(point.mean) / curveMax * 100)}%` }} /></div>
                  <strong>{decimal(point.mean, 4)}</strong>
                </div>
              )) : <div className="empty-evaluation-chart"><FlaskConical size={21} /><span>{blocked ? '检验通过后显示期限读数' : '首跑后显示 1 / 3 / 5 / 10 / 20 日读数'}</span></div>}
              <p className="overlap-warning"><AlertTriangle size={13} />N 日前瞻收益存在重叠；框架强制使用 lag ≥ N 的 Newey–West 推断。</p>
            </section>
          </div>

          <section className="evaluation-card build-card">
            <div className="evaluation-card-heading">
              <div><span>BUILD STATUS</span><h2>P0 建设面</h2></div>
              <small>{framework.contract.output_contract.implemented.length} 已落地 · {framework.contract.output_contract.next.length} 待续建</small>
            </div>
            <div className="build-columns">
              <div>
                <h3>本轮已落地</h3>
                {framework.contract.output_contract.implemented.map((item) => <p key={item}><Check size={13} />{labels[item] ?? item}</p>)}
              </div>
              <div>
                <h3>下一批（不隐式宣称完成）</h3>
                {framework.contract.output_contract.next.map((item) => <p key={item}><CircleDashed size={13} />{labels[item] ?? item}</p>)}
              </div>
            </div>
            <div className={`construction-block ${framework.contract.construction_gates.T3_next_day_execution_filter.status === 'completed' ? 'complete' : ''}`}>
              {framework.contract.construction_gates.T3_next_day_execution_filter.status === 'completed' ? <ShieldCheck size={14} /> : <LockKeyhole size={14} />}
              <div>
                <strong>{framework.contract.construction_gates.T3_next_day_execution_filter.status === 'completed' ? 'T3 过滤诊断已完成' : 'T3 进行中'}</strong>
                <span>{framework.contract.construction_gates.T3_next_day_execution_filter.execution_date_rule ?? framework.contract.construction_gates.T3_next_day_execution_filter.unblocked_by}</span>
              </div>
            </div>
          </section>
        </>
      )}
    </main>
  )
}

type ControlValue = IcSummaryControl | undefined
type IcSummaryControl = { passed: boolean; absolute_mean: number; absolute_mean_limit: number }
type ShiftedControl = { declined: boolean; baseline_mean_ic: number; shifted_mean_ic: number; decline_t: number; status: string }

function SimpleGateRow({ title, note, passed, detail }: { title: string; note: string; passed?: boolean; detail?: string }) {
  const ready = passed !== undefined
  return (
    <div className="control-row">
      <span className={ready ? (passed ? 'ok' : 'warn') : 'idle'}>{ready ? (passed ? <Check size={14} /> : <AlertTriangle size={14} />) : <CircleDashed size={14} />}</span>
      <div><strong>{title}</strong><small>{note}</small></div>
      <b>{detail ?? '—'}</b>
    </div>
  )
}

function ControlRow({ title, note, value, shifted }: { title: string; note: string; value?: ControlValue; shifted?: ShiftedControl }) {
  const passed = value?.passed ?? shifted?.declined
  const ready = Boolean(value || shifted)
  return (
    <div className="control-row">
      <span className={ready ? (passed ? 'ok' : 'warn') : 'idle'}>{ready ? (passed ? <Check size={14} /> : <AlertTriangle size={14} />) : <CircleDashed size={14} />}</span>
      <div><strong>{title}</strong><small>{note}</small></div>
      <b>{value ? `${decimal(value.absolute_mean, 4)} / ${decimal(value.absolute_mean_limit, 4)}` : shifted ? `${decimal(shifted.baseline_mean_ic, 4)} → ${decimal(shifted.shifted_mean_ic, 4)}` : '—'}</b>
    </div>
  )
}
