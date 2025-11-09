import { Clock, Trash2, RotateCcw } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { getDemoById, getCategoryById } from '../demos/demoRegistry';
import { formatTime } from '../utils/imageUtils';

export default function History() {
  const { history, restoreFromHistory, clearHistory } = useAppStore();

  if (history.length === 0) {
    return (
      <div className="card bg-dark text-light">
        <div className="card-header bg-secondary">
          <h3 className="h5 mb-0 d-flex align-items-center gap-2">
            <Clock size={20} />
            Result History
          </h3>
        </div>
        <div className="card-body">
          <div className="text-center text-light py-4">
            <p>No history yet. Process some images to build up history.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card bg-dark text-light">
      <div className="card-header bg-secondary">
        <div className="d-flex justify-content-between align-items-center">
          <h3 className="h5 mb-0 d-flex align-items-center gap-2">
            <Clock size={20} />
            Result History ({history.length})
          </h3>
          <button
            className="btn btn-sm btn-danger"
            onClick={clearHistory}
            title="Clear history"
          >
            <Trash2 size={14} />
            <span className="ms-1">Clear</span>
          </button>
        </div>
      </div>

      <div className="card-body">
        <div className="row g-2">
          {history.map((item) => {
            const demo = getDemoById(item.demo);
            const category = getCategoryById(item.category);

            return (
              <div key={item.id} className="col-12">
                <div
                  className="card bg-secondary text-light position-relative"
                  onClick={() => restoreFromHistory(item)}
                  title="Click to restore this result"
                  style={{ cursor: 'pointer' }}
                >
                  <div className="card-body p-2">
                    <div className="row g-2">
                      {item.outputThumbnail && (
                        <div className="col-auto">
                          <img
                            src={item.outputThumbnail}
                            alt="Result"
                            className="rounded"
                            style={{ width: '80px', height: '80px', objectFit: 'cover' }}
                          />
                        </div>
                      )}
                      <div className="col">
                        <div className="fw-bold text-light">{demo?.name}</div>
                        <div className="text-muted small">{category?.name}</div>
                        <div className="d-flex justify-content-between align-items-center mt-1">
                          <span className="badge bg-primary">
                            {item.processingTime ? formatTime(item.processingTime) : 'N/A'}
                          </span>
                          <span className="text-muted small">{item.timestamp}</span>
                        </div>
                        {item.params && Object.keys(item.params).length > 0 && (
                          <div className="d-flex gap-1 mt-2 flex-wrap">
                            {Object.entries(item.params).slice(0, 2).map(([key, value]) => (
                              <span key={key} className="badge bg-dark">
                                {key}: {typeof value === 'number' ? value.toFixed(1) : value}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="col-auto d-flex align-items-center">
                        <button
                          className="btn btn-sm btn-outline-primary"
                          title="Restore this result"
                        >
                          <RotateCcw size={16} />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
