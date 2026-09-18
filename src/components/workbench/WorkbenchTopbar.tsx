import { ChartNoAxesCombined, Database, Menu, Plus, RefreshCw, ShieldCheck, X } from 'lucide-react'
import { workbenchCopy, workbenchTabMeta, WorkbenchTab } from '../../config/workbenchConfig'

type WorkbenchTopbarProps = {
  openTabs: WorkbenchTab[]
  activeTab: WorkbenchTab
  onSelectTab: (tab: WorkbenchTab) => void
  onCloseTab: (tab: WorkbenchTab) => void
  onAddTab: () => void
  onOpenPipeline: () => void
  onRefreshAll: () => void
  onOpenToolbox?: () => void
  refreshing: boolean
}

export function WorkbenchTopbar({
  openTabs,
  activeTab,
  onSelectTab,
  onCloseTab,
  onAddTab,
  onOpenPipeline,
  onRefreshAll,
  onOpenToolbox,
  refreshing,
}: WorkbenchTopbarProps) {
  return (
    <div className="workbench-topbar">
      {onOpenToolbox ? (
        <button
          className="workbench-toolbox-trigger"
          onClick={onOpenToolbox}
          aria-label={workbenchCopy.sidebar.toolboxOpenTitle}
          title={workbenchCopy.sidebar.toolboxOpenTitle}
        >
          <Menu size={18} />
          <span>{workbenchCopy.sidebar.toolboxTitle}</span>
        </button>
      ) : null}
      <button className="workbench-home" aria-label={workbenchCopy.topbar.homeLabel}>
        <ShieldCheck size={20} />
        <strong>{workbenchCopy.topbar.homeLabel}</strong>
      </button>
      <div className="workspace-tabs">
        {openTabs.map((tab) => {
          const meta = workbenchTabMeta[tab]
          const Icon = meta.icon
          return (
            <div key={tab} className={`workspace-tab ${activeTab === tab ? 'active' : ''}`} onClick={() => onSelectTab(tab)}>
              <Icon size={16} />
              <span>{meta.label}</span>
                <button
                  onClick={(event) => {
                    event.stopPropagation()
                    onCloseTab(tab)
                  }}
                  aria-label={`${workbenchCopy.topbar.removeTabAriaPrefix}${meta.label}${workbenchCopy.topbar.removeTabAriaSuffix}`}
                >
                <X size={15} />
              </button>
            </div>
          )
        })}
        <button className="add-workspace" onClick={onAddTab} aria-label={workbenchCopy.topbar.addTab}>
          <Plus size={20} />
        </button>
      </div>
      <div className="workbench-global">
        <button onClick={() => { window.location.hash = '/chart-preview' }} title="回测图表 · 模拟数据预览" aria-label="打开模拟图表预览">
          <ChartNoAxesCombined size={16} />图表预览
        </button>
        <button onClick={onOpenPipeline}>
          <Database size={16} />
          {workbenchCopy.topbar.pipelineButton}
        </button>
        <button className={refreshing ? 'spinning' : ''} onClick={onRefreshAll}>
          <RefreshCw size={16} />
          {workbenchCopy.topbar.allRefresh}
        </button>
        <div className="workbench-avatar">{workbenchCopy.topbar.searchHint.slice(0, 1)}</div>
      </div>
    </div>
  )
}
