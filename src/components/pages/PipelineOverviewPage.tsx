import { Check, GitBranch, ShieldCheck, Database, ArrowRight } from 'lucide-react'
import { pipelineUiCopy } from '../../config/pipelineConfig'
import { StatusPill } from '../StatusPill'
import type { PipelineSummary, Snapshot } from '../../types'

type PipelineOverviewProps = {
  snapshot: Snapshot | null
  pipeline: PipelineSummary | null
}

const pipelineFlowNodes = pipelineUiCopy.overview.card.pipelineNode.nodes

export function PipelineOverviewPage({ snapshot, pipeline }: PipelineOverviewProps) {
  const blockers = snapshot ? Object.values(snapshot.quality_checks).reduce((sum, value) => sum + value, 0) : null
  const history = pipeline?.history

  return (
    <>
      <section className="hero">
        <div>
          <div className="eyebrow">
            <GitBranch size={14} />
            {pipelineUiCopy.overview.eyebrow}
          </div>
          <h1>{pipelineUiCopy.overview.title}</h1>
          <p>{pipelineUiCopy.overview.description}</p>
        </div>
        <StatusPill>{pipelineUiCopy.overview.marketBadge}</StatusPill>
      </section>
      <div className="metric-grid">
        <article className="metric-card">
          <div className="metric-top">
            <span>{pipelineUiCopy.overview.card.tradingDays.title}</span>
            <Check size={16} />
          </div>
          <strong>{history?.trading_days?.toLocaleString() ?? pipelineUiCopy.overview.card.tradingDays.valueMissing}</strong>
          <small>{history?.min_trade_date ?? '—'} → {history?.max_trade_date ?? '—'}</small>
        </article>
        <article className="metric-card">
          <div className="metric-top">
            <span>{pipelineUiCopy.overview.card.marketQuote.title}</span>
            <Database size={16} />
          </div>
          <strong>{history?.price_rows?.toLocaleString() ?? pipelineUiCopy.overview.card.marketQuote.valueMissing}</strong>
          <small>{history?.assets?.toLocaleString() ?? '—'}{pipelineUiCopy.overview.card.marketQuote.unit}</small>
        </article>
        <article className="metric-card">
          <div className="metric-top">
            <span>{pipelineUiCopy.overview.card.gating.title}</span>
            <ShieldCheck size={16} />
          </div>
          <strong>{blockers === null ? pipelineUiCopy.overview.card.gating.missing : `${blockers}${pipelineUiCopy.overview.card.gating.suffix}`}</strong>
            <small>{snapshot?.as_of_date ?? pipelineUiCopy.overview.card.gating.fallbackDate} {pipelineUiCopy.overview.card.pipelineNode.detailSuffix}</small>
        </article>
        <article className="metric-card dark-card">
          <div className="metric-top">
            <span>{pipelineUiCopy.overview.card.boundary.title}</span>
            <ArrowRight size={16} />
          </div>
          <strong>{pipelineUiCopy.overview.card.boundary.title}</strong>
            <small>{pipelineUiCopy.overview.card.boundary.title}{pipelineUiCopy.overview.card.pipelineNode.boundarySuffix}</small>
          <div className="manifest">{pipelineUiCopy.overview.card.boundary.badge}</div>
        </article>
      </div>
      <section className="panel pipeline-panel">
        <div className="panel-heading">
          <div>
            <h2>{pipelineUiCopy.overview.card.pipelineNode.subtitle}</h2>
            <p>{pipelineUiCopy.overview.card.pipelineNode.subtitle}</p>
          </div>
        </div>
        <div className="pipeline-flow">
          {pipelineFlowNodes.map((item, index) => (
            <div className="flow-wrap" key={item[0]}>
              <div className="flow-node green">
                <div className="node-icon"><Database size={18} /></div>
                <div>
                  <strong>{item[0]}</strong>
                  <small>{item[1]}</small>
                </div>
                <Check size={15} />
              </div>
              {index < pipelineFlowNodes.length - 1 && <ArrowRight className="flow-arrow" size={16} />}
            </div>
          ))}
        </div>
      </section>
    </>
  )
}
