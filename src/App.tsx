import { useEffect, useMemo, useState } from 'react'
import { CircleAlert } from 'lucide-react'
import { pipelineUiCopy } from './config/pipelineConfig'
import { PipelineOverviewPage } from './components/pages/PipelineOverviewPage'
import { QualityPage } from './components/pages/QualityPage'
import { SourcesPage } from './components/pages/SourcesPage'
import { Shell } from './components/Shell'
import { FactorWorkbench } from './components/workbench/FactorWorkbench'
import type { PipelineSummary, Snapshot, Workspace, Page } from './types'
import { BacktestPreviewPage } from './components/backtest/BacktestPreviewPage'

export default function App() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const update = () => setHash(window.location.hash)
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  // Anchor navigation inside the preview must not unmount it.
  return hash === '#/chart-preview' || hash.startsWith('#bp-')
    ? <BacktestPreviewPage /> : <WorkspaceApp />
}

function WorkspaceApp() {
  const initial: Workspace = window.location.hash.startsWith('#/pipeline') ? 'pipeline' : 'research'
  const [workspace, setWorkspaceState] = useState<Workspace>(initial)
  const [page, setPage] = useState<Page>('overview')
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [pipeline, setPipeline] = useState<PipelineSummary | null>(null)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    fetch('/api/research/snapshot')
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json()
      })
      .then(setSnapshot)
      .catch((error) => setLoadError(String(error)))
  }, [])

  useEffect(() => {
    fetch('/api/pipeline/summary')
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json()
      })
      .then(setPipeline)
      .catch((error) => setLoadError(String(error)))
  }, [])

  const setWorkspace = (value: Workspace) => {
    setWorkspaceState(value)
    setPage('overview')
    window.location.hash = `/${value}`
  }

  const content = useMemo(() => {
    if (loadError && !snapshot && !pipeline) {
      return <section className="panel loading-state"><CircleAlert size={20} /><strong>{pipelineUiCopy.loadingErrorPrefix}{loadError}</strong></section>
    }
    return page === 'quality'
      ? <QualityPage snapshot={snapshot} />
      : page === 'sources'
        ? <SourcesPage pipeline={pipeline} />
        : <PipelineOverviewPage snapshot={snapshot} pipeline={pipeline} />
  }, [page, snapshot, pipeline, loadError])

  if (workspace === 'research') {
    return <FactorWorkbench snapshot={snapshot} onOpenPipeline={() => setWorkspace('pipeline')} />
  }

  return (
    <Shell workspace={workspace} setWorkspace={setWorkspace} page={page} setPage={setPage}>
      {content}
    </Shell>
  )
}
