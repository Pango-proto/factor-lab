import type { ReactNode } from 'react'

export function WorkspaceHeader({
  eyebrow,
  title,
  description,
  aside,
  className = '',
}: {
  eyebrow: ReactNode
  title: ReactNode
  description?: ReactNode
  aside?: ReactNode
  className?: string
}) {
  return <div className={`workbench-title ${className}`.trim()}>
    <div><span>{eyebrow}</span><h1>{title}</h1>{description && <p>{description}</p>}</div>
    {aside}
  </div>
}
