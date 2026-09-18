import { Settings2, Search, Menu, GitBranch, Boxes, X } from 'lucide-react'
import { useMemo } from 'react'
import { draftCopy } from '../config/workbenchConfig'
import { pipelineNav } from '../config/pipelineConfig'
import type { Page, Workspace } from '../types'

type ShellProps = {
  workspace: Workspace
  setWorkspace: (value: Workspace) => void
  page: Page
  setPage: (value: Page) => void
  children: React.ReactNode
}

export function Shell({ workspace, setWorkspace, page, setPage, children }: ShellProps) {
  const currentPageLabel = useMemo(() => pipelineNav.find((item) => item.id === page)?.label ?? '', [page])

  return (
    <div className={`app-shell workspace-${workspace}`}>
      <aside>
        <div className="brand">
          <div className="brand-mark">
            {workspace === 'pipeline' ? <GitBranch size={19} /> : <Boxes size={19} />}
          </div>
          <div>
            <strong>{workspace === 'pipeline' ? draftCopy.brandTitle.pipeline : draftCopy.brandTitle.research}</strong>
            <span>{workspace === 'pipeline' ? draftCopy.brandDesc.pipeline : draftCopy.brandDesc.research}</span>
          </div>
          <button className="sidebar-close" onClick={() => {}} aria-label={draftCopy.shell.closeMenu}><X size={18} /></button>
        </div>
        <div className="workspace-switch">
          <button className={workspace === 'pipeline' ? 'active' : ''} onClick={() => setWorkspace('pipeline')}>
            <GitBranch size={14} />
            {draftCopy.shell.workspacePipeline}
          </button>
          <button className={workspace === 'research' ? 'active' : ''} onClick={() => setWorkspace('research')}>
            <Boxes size={14} />
            {draftCopy.shell.workspaceResearch}
          </button>
        </div>
        <nav>
          {pipelineNav.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)}>
                <Icon size={17} />
                {item.label}
              </button>
            )
          })}
        </nav>
        <div className="sidebar-bottom">
            <div className="env">
              <i />
              <div>
                <strong>{draftCopy.shell.localEnvironmentLabel}</strong>
                <span>{workspace === 'pipeline' ? draftCopy.boardEnvText.pipeline : draftCopy.boardEnvText.research}</span>
              </div>
          </div>
          <button aria-label={draftCopy.shell.settings}><Settings2 size={17} /></button>
        </div>
      </aside>
      <div className="main-shell">
        <header>
          <button className="menu-button" onClick={() => {}} aria-label={draftCopy.shell.openMenu}>
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            {workspace === 'pipeline' ? draftCopy.shell.workspacePipeline : draftCopy.shell.workspaceResearch}
            <span>/</span>
            {currentPageLabel}
          </div>
          <div className="header-actions">
            <button aria-label={draftCopy.shell.searchAria}><Search size={18} /></button>
            <div className="system-state">
              <i />
              {draftCopy.shell.localState}
            </div>
            <div className="avatar">{draftCopy.avatarText}</div>
          </div>
        </header>
        <main>{children}</main>
      </div>
    </div>
  )
}
