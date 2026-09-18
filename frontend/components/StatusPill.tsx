export function StatusPill({ children, tone = 'green' }: { children: React.ReactNode; tone?: 'green' | 'amber' }) {
  return <span className={`status-pill ${tone}`}><span />{children}</span>
}
