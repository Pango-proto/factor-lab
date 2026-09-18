import { Check, CircleAlert } from 'lucide-react'
import { pipelineUiCopy } from '../../config/pipelineConfig'
import { StatusPill } from '../StatusPill'
import type { Snapshot } from '../../types'

type QualityPageProps = { snapshot: Snapshot | null }

export function QualityPage({ snapshot }: QualityPageProps) {
  const checks = snapshot
    ? [
      [pipelineUiCopy.quality.checkTitles[0], snapshot.quality_checks.duplicate_assets],
      [pipelineUiCopy.quality.checkTitles[1], snapshot.quality_checks.future_available_values],
      [pipelineUiCopy.quality.checkTitles[2], snapshot.quality_checks.missing_market_cap],
    ] as const
    : []

  const blockers = checks.reduce((sum, [, value]) => sum + value, 0)

  return (
    <>
      <section className="page-title">
        <div>
          <span>{pipelineUiCopy.quality.subtitle}</span>
          <h1>{pipelineUiCopy.quality.title}</h1>
          <p>{pipelineUiCopy.quality.intro}</p>
        </div>
        <StatusPill tone={blockers === 0 ? 'green' : 'amber'}>
          {snapshot
            ? `${blockers === 0 ? pipelineUiCopy.quality.statusCopy.pass : pipelineUiCopy.quality.statusCopy.fail} · ${blockers}${pipelineUiCopy.quality.statusCopy.blockerSuffix}`
            : pipelineUiCopy.quality.statusCopy.waiting}
        </StatusPill>
      </section>
      <section className="panel checklist">
        <span className="card-label">{pipelineUiCopy.quality.checksTitle}</span>
        {checks.map(([label, value]) => (
          <div key={label}>
            {value === 0 ? <Check size={15} /> : <CircleAlert size={15} />}
            {label}
            <b>{value}{pipelineUiCopy.quality.statusCopy.issueSuffix}</b>
          </div>
        ))}
      </section>
      <div className="note-card">
        <CircleAlert size={20} />
        <div>
          <strong>{pipelineUiCopy.quality.noteTitle}</strong>
          <p>{pipelineUiCopy.quality.note}</p>
        </div>
      </div>
    </>
  )
}
