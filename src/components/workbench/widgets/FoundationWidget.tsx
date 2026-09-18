import { CircleAlert, Database, ShieldCheck } from 'lucide-react'
import { workbenchCopy } from '../../../config/workbenchConfig'
import type { Snapshot } from '../../../types'

type FoundationWidgetProps = { snapshot: Snapshot | null }

export function FoundationWidget({ snapshot }: FoundationWidgetProps) {
  const blockers = snapshot ? Object.values(snapshot.quality_checks).reduce((sum, value) => sum + value, 0) : '—'

  return (
    <>
      <div className="foundation-gates">
        <section className="ready">
          <span>{workbenchCopy.foundation.noData.matrixLabel}</span>
          <Database size={21} />
          <div>
            <strong>{workbenchCopy.foundation.noData.matrixTitle}</strong>
            <p>
              {snapshot?.counts.eligible_market_matrix.toLocaleString() ?? '—'}
              {workbenchCopy.foundation.noData.sampleCountSuffix}
            </p>
          </div>
          <b>{workbenchCopy.foundation.noData.matrixStatus}</b>
        </section>
        <section className="ready">
          <span>{workbenchCopy.foundation.noData.qualityLabel}</span>
          <ShieldCheck size={21} />
          <div>
            <strong>{workbenchCopy.foundation.noData.qualityTitle}</strong>
            <p>{blockers}{workbenchCopy.foundation.noData.blockersSuffix}</p>
          </div>
          <b>{workbenchCopy.foundation.noData.qualityStatusText}</b>
        </section>
      </div>
      <div className="roadmap-redline">
        <CircleAlert size={15} />
        <span>{workbenchCopy.foundation.noData.noFactorTag}</span>
      </div>
    </>
  )
}
