import { Activity, Database, Library, Plus, RefreshCw, X } from 'lucide-react'
import { workbenchTools, workbenchTabMeta, WorkbenchTab } from '../../config/workbenchConfig'
import { workbenchCopy } from '../../config/workbenchConfig'

type WorkbenchSidebarProps = {
  isOpen: boolean
  activeTab: WorkbenchTab
  onOpenTab: (tab: WorkbenchTab) => void
  onOpenPipeline: () => void
  onRefreshRegistry: () => void
  onOpenRegistry: () => void
  onOpenLibrary: () => void
  onToggleOpen: () => void
}

export function WorkbenchSidebar({
  isOpen,
  activeTab,
  onOpenTab,
  onOpenPipeline,
  onRefreshRegistry,
  onOpenRegistry,
  onOpenLibrary,
  onToggleOpen,
}: WorkbenchSidebarProps) {
  return (
    <aside
      id="research-tool-drawer"
      className={`tool-drawer${isOpen ? ' open' : ''}`}
    >
      <div className="tool-drawer-head">
        <div>
          <span>{workbenchCopy.sidebar.toolboxTitle}</span>
          <h2>{workbenchCopy.sidebar.title}</h2>
        </div>
        <button
          onClick={onToggleOpen}
          aria-label={workbenchCopy.sidebar.toolboxCloseTitle}
          title={workbenchCopy.sidebar.toolboxCloseTitle}
        >
          <X size={18} />
        </button>
      </div>
      <nav className="tool-workflow">
        <strong>{workbenchCopy.toolbar.workflowTitle}</strong>
        {workbenchTools.map((item, index) => {
          const Icon = workbenchTabMeta[item.id].icon
          return (
            <button
              key={item.id}
              className={activeTab === item.id ? 'active' : ''}
              onClick={() => onOpenTab(item.id)}
            >
              <span>{String(index + 1).padStart(2, '0')}</span>
              <Icon size={17} />
              <div>
                <b>{item.label}</b>
                <small>{item.note}</small>
              </div>
              <em className={item.status}>
                {item.status === 'ready' ? workbenchCopy.locked.draftState.ready : workbenchCopy.locked.draftState.planned}
              </em>
            </button>
          )
        })}
      </nav>
      <div className="tool-quick-actions">
        <strong>{workbenchCopy.toolbar.quickTitle}</strong>
        <button onClick={onOpenPipeline}><Database size={16} /><span>{workbenchCopy.toolbar.quickActions.openPipeline}</span></button>
        <button onClick={onRefreshRegistry}><RefreshCw size={16} /><span>{workbenchCopy.toolbar.quickActions.refreshRegistry}</span></button>
        <button onClick={onOpenRegistry}><Library size={16} /><span>{workbenchCopy.toolbar.quickActions.openRegistry}</span></button>
        <button onClick={onOpenLibrary}><Plus size={16} /><span>{workbenchCopy.toolbar.quickActions.addWidget}</span></button>
      </div>
      <div className="tool-drawer-foot">
        <Activity size={15} />
        <span>{workbenchCopy.toolbar.statusText}</span>
      </div>
    </aside>
  )
}
