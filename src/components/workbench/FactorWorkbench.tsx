import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Library, Plus } from 'lucide-react'
import { BoardId, FactorCatalogResult, Snapshot } from '../../types'
import {
  boardTabs,
  componentCatalog,
  ComponentId,
  defaultOpenTabs,
  defaultWidgetSizes,
  defaultWorkbenchComponents,
  workbenchCopy,
  WorkbenchTab,
  workbenchTabOrder,
} from '../../config/workbenchConfig'
import { LockedWorkspace } from './locked/LockedWorkspace'
import { FactorLabWidget } from './widgets/FactorLabWidget'
import { FoundationWidget } from './widgets/FoundationWidget'
import { WorkbenchTopbar } from './WorkbenchTopbar'
import { WorkbenchSidebar } from './WorkbenchSidebar'
import { WorkbenchWidget } from './WorkbenchWidget'
import { ComponentLibraryModal } from './ComponentLibraryModal'
import { WorkspaceHeader } from '../WorkspaceHeader'
import { EvaluationFrameworkPage } from './EvaluationFrameworkPage'

type FactorWorkbenchProps = {
  snapshot: Snapshot | null
  onOpenPipeline: () => void
}

type FactorFamily = 'alpha' | 'risk'

export function FactorWorkbench({ snapshot, onOpenPipeline }: FactorWorkbenchProps) {
  const [selectedBoard, setSelectedBoard] = useState<BoardId>('MAIN')
  const [factorQuery, setFactorQuery] = useState('')
  const [factorFamily, setFactorFamily] = useState<FactorFamily>('alpha')
  const [factorCatalog, setFactorCatalog] = useState<FactorCatalogResult | null>(null)
  const [error, setError] = useState('')
  const [openTabs, setOpenTabs] = useState<WorkbenchTab[]>(defaultOpenTabs)
  const [activeTab, setActiveTab] = useState<WorkbenchTab>('strategy')
  const [components, setComponents] = useState<ComponentId[]>(() => {
    try {
      const saved = localStorage.getItem('factor-workbench-components')
      const parsed = saved ? JSON.parse(saved) : null
      if (!Array.isArray(parsed)) return defaultWorkbenchComponents
      return parsed.filter((id): id is ComponentId =>
        componentCatalog.some((item) => item.id === id),
      )
    } catch {
      return defaultWorkbenchComponents
    }
  })
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [isToolDrawerOpen, setIsToolDrawerOpen] = useState(true)
  const [librarySearch, setLibrarySearch] = useState('')
  const [dragged, setDragged] = useState<ComponentId | null>(null)
  const [sizes, setSizes] = useState<Record<ComponentId, { columns: number; rows: number }>>(() => {
    try {
      const saved = localStorage.getItem('factor-workbench-sizes')
      const parsed = saved ? JSON.parse(saved) : null
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return defaultWidgetSizes
      const valid: Record<ComponentId, { columns: number; rows: number }> = {} as Record<ComponentId, { columns: number; rows: number }>
      for (const [id, size] of Object.entries(parsed) as Array<[ComponentId, unknown]>) {
        if (!componentCatalog.some((item) => item.id === id)) continue
        if (typeof size !== 'object' || size === null) continue
        const candidate = size as { columns?: unknown; rows?: unknown }
        const columns = Number(candidate.columns)
        const rows = Number(candidate.rows)
        if (!Number.isFinite(columns) || !Number.isFinite(rows)) continue
        valid[id] = { columns, rows }
      }
      return {
        ...defaultWidgetSizes,
        ...valid,
      }
    } catch {
      return defaultWidgetSizes
    }
  })
  const dashboardRef = useRef<HTMLDivElement | null>(null)
  const [refreshing, setRefreshing] = useState<ComponentId | 'all' | null>(null)

  const loadFactorCatalog = async (query: string) => {
    const normalized = query.trim()
    if (!normalized) return setFactorCatalog(null)
    try {
      const response = await fetch(`/api/factors?q=${encodeURIComponent(normalized)}&family=${factorFamily}&page=1&page_size=20`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setFactorCatalog(await response.json())
      setError('')
    } catch (reason) {
      setError(`${workbenchCopy.errors.factorRegistryLoadFailed}${String(reason)}`)
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadFactorCatalog(factorQuery)
    }, 220)
    return () => window.clearTimeout(timer)
  }, [factorQuery, factorFamily])

  useEffect(() => {
    localStorage.setItem('factor-workbench-components', JSON.stringify(components))
  }, [components])

  useEffect(() => {
    localStorage.setItem('factor-workbench-sizes', JSON.stringify(sizes))
  }, [sizes])

  const closeTab = (tab: WorkbenchTab) => {
    setOpenTabs((current) => {
      const next = current.filter((item) => item !== tab)
      if (activeTab === tab && next.length) {
        const currentIndex = current.indexOf(tab)
        setActiveTab(next[Math.max(0, currentIndex - 1)])
      }
      return next
    })
  }

  const openWorkbenchTab = (tab: WorkbenchTab) => {
    setOpenTabs((current) => (current.includes(tab) ? current : workbenchTabOrder.filter((item) => current.includes(item) || item === tab)))
    setActiveTab(tab)
  }

  const addNextTab = () => {
    const next = workbenchTabOrder.find((tab) => !openTabs.includes(tab))
    if (next) {
      setOpenTabs((current) => workbenchTabOrder.filter((item) => current.includes(item) || item === next))
      setActiveTab(next)
    }
  }

  const removeComponent = (id: ComponentId) => setComponents((current) => current.filter((item) => item !== id))
  const addComponent = (id: ComponentId) => setComponents((current) => (current.includes(id) ? current : [...current, id]))
  const dropComponent = (target: ComponentId) => {
    if (!dragged || dragged === target) return setDragged(null)
    setComponents((current) => {
      const next = current.filter((item) => item !== dragged)
      next.splice(next.indexOf(target), 0, dragged)
      return next
    })
    setDragged(null)
  }

  const startResize = (event: ReactPointerEvent<HTMLButtonElement>, id: ComponentId) => {
    event.preventDefault()
    event.stopPropagation()
    const dashboard = dashboardRef.current
    if (!dashboard) return
    const startX = event.clientX
    const startY = event.clientY
    const initial = sizes[id]
    const columnWidth = dashboard.getBoundingClientRect().width / 12
    const onMove = (moveEvent: PointerEvent) => {
      const columns = Math.max(3, Math.min(12, Math.round(initial.columns + (moveEvent.clientX - startX) / columnWidth)))
      const rows = Math.max(8, Math.min(48, Math.round(initial.rows + (moveEvent.clientY - startY) / 22)))
      setSizes((current) => ({ ...current, [id]: { columns, rows } }))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.classList.remove('is-resizing-widget')
    }
    document.body.classList.add('is-resizing-widget')
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
  }

  const refreshComponent = async (id: ComponentId) => {
    setRefreshing(id)
    if (id === 'factorLab' && factorQuery.trim()) await loadFactorCatalog(factorQuery)
    window.setTimeout(() => setRefreshing(null), 250)
  }

  const refreshAll = () => {
    setRefreshing('all')
    if (factorQuery.trim()) void loadFactorCatalog(factorQuery)
    window.setTimeout(() => setRefreshing(null), 250)
  }

  const strategyWorkspace = (
    <main className="workbench-main">
      <div className="board-tabs" role="tablist" aria-label={workbenchCopy.sidebar.boardAriaLabel}>
        {boardTabs.map((board) => (
          <button
            role="tab"
            aria-selected={selectedBoard === board.id}
            className={selectedBoard === board.id ? 'active' : ''}
            key={board.id}
            onClick={() => setSelectedBoard(board.id)}
          >
            <strong>{board.label}</strong>
            <span>{snapshot?.counts.by_board?.[board.id]?.eligible_market_matrix?.toLocaleString() ?? '—'} {workbenchCopy.sidebar.boardSuffix}</span>
          </button>
        ))}
      </div>
      <WorkspaceHeader
        eyebrow={`${workbenchCopy.boardTabs.titlePrefix}${selectedBoard}`}
        title={`${boardTabs.find((board) => board.id === selectedBoard)?.label}${workbenchCopy.workbenchStates.strategy.title}`}
        description={workbenchCopy.workbenchStates.strategy.subtitle}
        aside={<div className="strategy-state"><i /><div><strong>{workbenchCopy.workbenchStates.strategy.warningTitle}</strong><span>{workbenchCopy.workbenchStates.strategy.warningNote}</span></div></div>}
      />
      {error ? <div className="workbench-error">{error}</div> : null}
      <div className="strategy-dashboard" ref={dashboardRef}>
        {components.map((id) => {
          const definition = componentCatalog.find((item) => item.id === id)
          if (!definition) return null
          const size = sizes[id]
          return (
            <WorkbenchWidget
              key={id}
              id={id}
              title={definition.title}
              subtitle={definition.subtitle}
              size={size}
              isWide={definition.size === 'wide'}
              isDragging={dragged === id}
              onDragStart={(widgetId) => setDragged(widgetId)}
              onDragEnd={() => setDragged(null)}
              onDrop={dropComponent}
              onResize={startResize}
              onRefresh={() => void refreshComponent(id)}
              onDelete={() => removeComponent(id)}
              isRefreshing={refreshing === id}
            >
              {id === 'foundation' ? (
                <FoundationWidget snapshot={snapshot} />
              ) : (
                <FactorLabWidget
                  selectedBoard={selectedBoard}
                  factorQuery={factorQuery}
                  factorFamily={factorFamily}
                  factorCatalog={factorCatalog}
                  errorText={error}
                  onFamilyChange={(value) => setFactorFamily(value)}
                  onQueryChange={setFactorQuery}
                />
              )}
            </WorkbenchWidget>
          )
        })}
      </div>
      {components.length === 0 && (
        <div className="empty-workbench">
          <h2>{workbenchCopy.locked.emptyWidgets.title}</h2>
          <p>{workbenchCopy.locked.emptyWidgets.description}</p>
          <button onClick={() => setLibraryOpen(true)}><Plus size={16} />{workbenchCopy.topbar.addComponent}</button>
        </div>
      )}
    </main>
  )

  const activeWorkspace = openTabs.length === 0
    ? (
      <main className="workbench-main">
        <div className="empty-workbench">
          <h2>{workbenchCopy.locked.noWorkspace.title}</h2>
          <p>{workbenchCopy.locked.noWorkspace.description}</p>
          <button onClick={addNextTab}><Plus size={16} />{workbenchCopy.locked.noWorkspace.action}</button>
        </div>
      </main>
    )
      : activeTab === 'factorReturns'
      ? <LockedWorkspace eyebrow={workbenchCopy.locked.factorReturns.eyebrow} title={workbenchCopy.locked.quality.title} description={workbenchCopy.locked.quality.description} />
      : activeTab === 'factorSets'
        ? <LockedWorkspace eyebrow={workbenchCopy.locked.factorSets.eyebrow} title={workbenchCopy.locked.factorSets.title} description={workbenchCopy.locked.factorSets.description} />
      : activeTab === 'scoring'
        ? <LockedWorkspace
          eyebrow={workbenchCopy.locked.scoring.eyebrow.replace('{board}', selectedBoard)}
          title={workbenchCopy.locked.scoring.title}
          description={workbenchCopy.locked.scoring.description}
        />
      : activeTab === 'optimizer'
        ? <LockedWorkspace eyebrow={workbenchCopy.locked.optimizer.eyebrow} title={workbenchCopy.locked.optimizer.title} description={workbenchCopy.locked.optimizer.description} />
      : activeTab === 'backtest'
        ? <EvaluationFrameworkPage />
      : activeTab === 'simulation'
        ? <LockedWorkspace eyebrow={workbenchCopy.locked.simulation.eyebrow} title={workbenchCopy.locked.simulation.title} description={workbenchCopy.locked.simulation.description} />
      : strategyWorkspace

  return (
    <div className={`factor-workbench-shell${isToolDrawerOpen ? '' : ' toolbox-collapsed'}`}>
      <WorkbenchSidebar
        isOpen={isToolDrawerOpen}
        activeTab={activeTab}
        onOpenTab={openWorkbenchTab}
        onToggleOpen={() => setIsToolDrawerOpen(false)}
        onOpenPipeline={onOpenPipeline}
        onRefreshRegistry={() => {
          if (factorQuery.trim()) void loadFactorCatalog(factorQuery)
        }}
        onOpenRegistry={() => {
          openWorkbenchTab('strategy')
          if (!components.includes('factorLab')) setComponents((current) => [...current, 'factorLab'])
        }}
        onOpenLibrary={() => setLibraryOpen(true)}
      />
      <div className="factor-workbench-main-area">
        <WorkbenchTopbar
          openTabs={openTabs}
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          onCloseTab={closeTab}
          onAddTab={addNextTab}
          onOpenPipeline={onOpenPipeline}
          onRefreshAll={refreshAll}
          onOpenToolbox={!isToolDrawerOpen ? () => setIsToolDrawerOpen(true) : undefined}
          refreshing={refreshing === 'all'}
        />
        {activeWorkspace}
        {activeTab === 'strategy' && (
          <>
            <button className="floating-library" onClick={() => setLibraryOpen(true)} aria-label={workbenchCopy.topbar.addComponent}>
              <Library size={21} />
            </button>
            <ComponentLibraryModal
              isOpen={libraryOpen}
              components={components}
              query={librarySearch}
              onClose={() => setLibraryOpen(false)}
              onQueryChange={setLibrarySearch}
              onToggleComponent={addComponent}
            />
          </>
        )}
      </div>
    </div>
  )
}
