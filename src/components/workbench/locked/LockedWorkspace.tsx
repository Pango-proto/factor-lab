import { WorkspaceHeader } from '../../WorkspaceHeader'

type LockedWorkspaceProps = {
  eyebrow: string
  title: string
  description: string
}

export function LockedWorkspace({ eyebrow, title, description }: LockedWorkspaceProps) {
  return (
    <main className="workbench-main">
      <WorkspaceHeader eyebrow={eyebrow} title={title} description={description} />
      <div className="empty-workbench">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </main>
  )
}
