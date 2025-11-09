import { Clock, Trash2, RotateCcw } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { getDemoById, getCategoryById } from '../demos/demoRegistry';
import { formatTime } from '../utils/imageUtils';

export default function History() {
  const { history, restoreFromHistory, clearHistory } = useAppStore();

  if (history.length === 0) {
    return (
      <div className="history-section">
        <div className="section-header">
          <h3>
            <Clock size={20} />
            Result History
          </h3>
        </div>
        <div className="history-placeholder">
          <p>No history yet. Process some images to build up history.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-section">
      <div className="section-header">
        <h3>
          <Clock size={20} />
          Result History ({history.length})
        </h3>
        <button
          className="btn btn-small btn-danger"
          onClick={clearHistory}
          title="Clear history"
        >
          <Trash2 size={14} />
          Clear
        </button>
      </div>

      <div className="history-grid">
        {history.map((item) => {
          const demo = getDemoById(item.demo);
          const category = getCategoryById(item.category);

          return (
            <div
              key={item.id}
              className="history-item"
              onClick={() => restoreFromHistory(item)}
              title="Click to restore this result"
            >
              {item.outputThumbnail && (
                <div className="history-thumbnail">
                  <img src={item.outputThumbnail} alt="Result" />
                </div>
              )}
              <div className="history-details">
                <div className="history-demo-name">{demo?.name}</div>
                <div className="history-category">{category?.name}</div>
                <div className="history-meta">
                  <span className="history-time">
                    {item.processingTime ? formatTime(item.processingTime) : 'N/A'}
                  </span>
                  <span className="history-timestamp">{item.timestamp}</span>
                </div>
                {item.params && Object.keys(item.params).length > 0 && (
                  <div className="history-params">
                    {Object.entries(item.params).slice(0, 2).map(([key, value]) => (
                      <span key={key} className="param-badge">
                        {key}: {typeof value === 'number' ? value.toFixed(1) : value}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <button
                className="history-restore-btn"
                title="Restore this result"
              >
                <RotateCcw size={16} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
