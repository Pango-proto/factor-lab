import { GripVertical, RefreshCw, Trash2 } from 'lucide-react'
import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react'
import type { WidgetSize, ComponentId } from '../../config/workbenchConfig'
import { workbenchCopy } from '../../config/workbenchConfig'

type WorkbenchWidgetProps = {
  id: ComponentId
  title: string
  subtitle: string
  size: WidgetSize
  isWide: boolean
  isDragging: boolean
  onDragStart: (id: ComponentId) => void
  onDragEnd: () => void
  onDrop: (targetId: ComponentId) => void
  onResize: (event: ReactPointerEvent<HTMLButtonElement>, id: ComponentId) => void
  onRefresh: () => void
  onDelete: () => void
  children: ReactNode
  isRefreshing: boolean
}

export function WorkbenchWidget({
  id,
  title,
  subtitle,
  size,
  isWide,
  isDragging,
  onDragStart,
  onDragEnd,
  onDrop,
  onResize,
  onRefresh,
  onDelete,
  children,
  isRefreshing,
}: WorkbenchWidgetProps) {
  return (
    <section
      className={`strategy-widget${isWide ? ' widget-wide' : ''}`}
      style={{ gridColumn: `span ${size.columns}`, gridRow: `span ${size.rows}` }}
      onDragOver={(event) => event.preventDefault()}
      onDrop={() => onDrop(id)}
    >
      <div className={`widget-heading ${isDragging ? 'dragging' : ''}`}>
        <button
          className="drag-handle"
          draggable
          onDragStart={() => onDragStart(id)}
          onDragEnd={() => onDragEnd()}
          aria-label={`${workbenchCopy.widget.dragAriaPrefix}${title}`}
          title={workbenchCopy.widget.dragLabel}
        >
          <GripVertical size={17} />
        </button>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <div className="widget-actions">
          <button className={isRefreshing ? 'spinning' : ''} onClick={onRefresh} aria-label={`${workbenchCopy.widget.refreshAriaPrefix}${title}`}>
            <RefreshCw size={16} />
          </button>
          <button onClick={onDelete} aria-label={`${workbenchCopy.widget.deleteAriaPrefix}${title}`}>
            <Trash2 size={16} />
          </button>
        </div>
      </div>
      {children}
      <button
        className="resize-handle"
        draggable={false}
        onPointerDown={(event) => onResize(event, id)}
        aria-label={`${workbenchCopy.widget.resizeAriaPrefix}${title}${workbenchCopy.widget.resizeAriaSuffix}`}
        title={workbenchCopy.widget.resizeLabel}
      />
    </section>
  )
}
