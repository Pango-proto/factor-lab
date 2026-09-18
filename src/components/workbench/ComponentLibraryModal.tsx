import { Search, X } from 'lucide-react'
import { componentCatalog } from '../../config/workbenchConfig'
import { workbenchCopy } from '../../config/workbenchConfig'
import { type ComponentId } from '../../config/workbenchConfig'

type ComponentLibraryModalProps = {
  isOpen: boolean
  components: ComponentId[]
  query: string
  onClose: () => void
  onQueryChange: (value: string) => void
  onToggleComponent: (id: ComponentId) => void
}

export function ComponentLibraryModal({
  isOpen,
  components,
  query,
  onClose,
  onQueryChange,
  onToggleComponent,
}: ComponentLibraryModalProps) {
  if (!isOpen) return null

  const keyword = query.toLowerCase()
  const available = componentCatalog.filter((item) => (item.title + item.subtitle).toLowerCase().includes(keyword))

  return (
      <div className="library-overlay" onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}>
      <div className="component-library">
        <div className="library-topbar">
          <div className="library-head">
            <div>
              <span>{workbenchCopy.componentLibrary.label}</span>
              <h2>{workbenchCopy.componentLibrary.title}</h2>
            </div>
            <button onClick={onClose} aria-label={workbenchCopy.componentLibrary.closeButton}>
              <X size={22} />
            </button>
          </div>
          <div className="library-search">
            <Search size={17} />
            <input value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder={workbenchCopy.componentLibrary.searchPlaceholder} />
          </div>
        </div>
          <div className="library-content">
            <div className="library-section-title">
              <h3>{workbenchCopy.componentLibrary.sectionTitle}</h3>
              <span>{available.length}</span>
            </div>
          <div className="library-grid">
            {available.map((item) => {
              const Icon = item.icon
              const isAdded = components.includes(item.id)
              return (
                <button
                  className={isAdded ? 'added' : ''}
                  key={item.id}
                  onClick={() => onToggleComponent(item.id)}
                >
                  <div className="component-preview">
                    <Icon size={31} />
                    <span>{workbenchCopy.componentLibrary.previewBadge}</span>
                  </div>
                  <strong>{item.title}</strong>
                  <p>{item.subtitle}</p>
                  <em>{isAdded ? workbenchCopy.componentLibrary.completeText : workbenchCopy.componentLibrary.addText}</em>
                </button>
              )
            })}
          </div>
        </div>
        <div className="library-footer">
          <span>
            {workbenchCopy.componentLibrary.footerLabel.replace('{count}', String(components.length))}
          </span>
          <button onClick={onClose}>{workbenchCopy.componentLibrary.doneButton}</button>
        </div>
      </div>
    </div>
  )
}
