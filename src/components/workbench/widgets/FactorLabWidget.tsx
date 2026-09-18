import { Search } from 'lucide-react'
import { factorFamilyOptions, factorRoleLabel, workbenchCopy } from '../../../config/workbenchConfig'
import type { BoardId, FactorCatalogResult } from '../../../types'

type FactorLabWidgetProps = {
  selectedBoard: BoardId
  factorQuery: string
  factorFamily: 'alpha' | 'risk'
  factorCatalog: FactorCatalogResult | null
  errorText: string
  onFamilyChange: (value: 'alpha' | 'risk') => void
  onQueryChange: (value: string) => void
}

export function FactorLabWidget({
  selectedBoard,
  factorQuery,
  factorFamily,
  factorCatalog,
  errorText,
  onFamilyChange,
  onQueryChange,
}: FactorLabWidgetProps) {
  const queryText = factorQuery.trim()
  const totalText = factorCatalog
    ? `${workbenchCopy.factorLab.placeholderSuffix.searchSummaryPrefix} ${factorCatalog.total} ${workbenchCopy.factorLab.placeholderSuffix.searchSummarySuffix} ${factorCatalog.page_size}`
    : `${selectedBoard} · ${workbenchCopy.factorLab.placeholderSuffix.waitingSearch}`

  return (
    <>
      <div className="factor-lab-search">
        <div className="factor-family-tabs">
          {factorFamilyOptions.map((family) => (
            <button
              key={family.value}
              className={factorFamily === family.value ? 'active' : ''}
              onClick={() => onFamilyChange(family.value)}
            >
              {family.label}
            </button>
          ))}
        </div>
        <div className="stock-search">
          <Search size={15} />
          <input value={factorQuery} onChange={(event) => onQueryChange(event.target.value)} placeholder={workbenchCopy.factorLab.searchPlaceholder} />
        </div>
        <span>{workbenchCopy.factorLab.sourceHint}</span>
      </div>
      {errorText ? <div className="workbench-error">{errorText}</div> : null}
      {!queryText ? (
        <div className="widget-loading">{workbenchCopy.factorLab.loadingHint}</div>
      ) : factorCatalog?.rows.length ? (
        <div className="factor-lab-table">
          <table>
            <thead>
              <tr>
                {workbenchCopy.factorLab.tableHeaders.map((head) => <th key={head}>{head}</th>)}
              </tr>
            </thead>
            <tbody>
              {factorCatalog.rows.map((factor) => {
                const pipeline = factor.family === 'risk'
                  ? (factor.orthogonalize_after.length ? `${workbenchCopy.factorLab.afterLabel}${factor.orthogonalize_after.join(', ')}` : workbenchCopy.factorLab.orthogonalizePrefix)
                  : `neutralize: ${factor.neutralize_against ?? 'none'} · risk set ${factor.risk_set_version ?? '—'}`
                return (
                  <tr key={`${factor.factor_id}@${factor.version}`}>
                    <td>
                      <strong>{factor.label}</strong>
                      <code>{factor.factor_id}@{factor.version}</code>
                      <small>{factor.formula}</small>
                    </td>
                    <td><span className="lab-status candidate">{factorRoleLabel[factor.role] ?? factor.role}</span></td>
                    <td>
                      <code>{pipeline}</code>
                      <small>{factor.pit_key} · coverage {factor.coverage_min ?? '—'}</small>
                    </td>
                    <td>
                      <span className="lab-status insufficient_data">{factor.research_status}{workbenchCopy.factorLab.forbidDeploySuffix}</span>
                      <small>{workbenchCopy.factorLab.variantPrefix}{factor.variant_count}</small>
                    </td>
                    <td><p>{factor.description}</p></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="widget-loading">{workbenchCopy.factorLab.emptyText}</div>
      )}
      <div className="widget-foot">
        <span>{totalText}</span>
        <b>{workbenchCopy.factorLab.bottom}</b>
      </div>
    </>
  )
}
