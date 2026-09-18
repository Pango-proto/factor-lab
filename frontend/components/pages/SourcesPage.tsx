import { pipelineUiCopy } from '../../config/pipelineConfig'
import { StatusPill } from '../StatusPill'
import type { PipelineSummary } from '../../types'

type SourcesPageProps = { pipeline: PipelineSummary | null }

export function SourcesPage({ pipeline }: SourcesPageProps) {
  const rows = pipeline?.tables ?? []

  return (
    <>
      <section className="page-title">
        <div>
          <span>{pipelineUiCopy.sources.catalogTitle}</span>
          <h1>{pipelineUiCopy.sources.title}</h1>
          <p>{pipelineUiCopy.sources.intro}</p>
        </div>
      </section>
      <section className="panel">
        <div className="data-table-wrap">
          <table>
            <thead>
              <tr>
                {pipelineUiCopy.sources.header.map((item) => <th key={item}>{item}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                  <tr key={row.id}>
                    <td><strong>{row.label}</strong></td>
                    <td><code>{row.id}</code></td>
                  <td>{row.rows.toLocaleString()} {pipelineUiCopy.sources.rowUnit}</td>
                  <td>
                    <StatusPill tone={row.status === 'complete' ? 'green' : 'amber'}>
                      {row.status === 'complete' ? pipelineUiCopy.sources.badge.complete : pipelineUiCopy.sources.badge.missing}
                    </StatusPill>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  )
}
