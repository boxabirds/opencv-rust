import { useState } from 'react';
import { ChevronRight, Check, Zap } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { categories, getDemosByCategory } from '../demos/demoRegistry';

export default function Sidebar() {
  const { selectedCategory, selectedDemo, selectDemo } = useAppStore();
  const [expandedCategories, setExpandedCategories] = useState(['filters', 'edges', 'transform', 'color']);

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev =>
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  return (
    <div className="h-100 d-flex flex-column">
      <div className="p-3 border-bottom border-dark sticky-top bg-secondary">
        <h2 className="h6 mb-1 text-light fw-bold">OpenCV Demos</h2>
        <p className="small text-muted mb-0">Interactive Image Processing</p>
      </div>

      <div className="flex-grow-1 overflow-auto p-2">
        {categories.map(category => {
          const demos = getDemosByCategory(category.id);
          const implementedCount = demos.filter(d => d.implemented).length;
          const isExpanded = expandedCategories.includes(category.id);

          return (
            <div key={category.id} className="mb-2">
              <button
                className="btn btn-sm btn-dark w-100 text-start d-flex align-items-center justify-content-between p-2"
                onClick={() => toggleCategory(category.id)}
              >
                <div className="d-flex align-items-center">
                  <ChevronRight
                    size={14}
                    className="me-2"
                    style={{ transform: isExpanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }}
                  />
                  <span className="small">{category.name}</span>
                </div>
                <span className="badge bg-primary">{implementedCount}/{demos.length}</span>
              </button>

              {isExpanded && (
                <div className="ms-3 mt-1">
                  {demos.map(demo => (
                    <button
                      key={demo.id}
                      className={`btn btn-sm w-100 text-start mb-1 p-2 ${
                        selectedCategory === category.id && selectedDemo === demo.id
                          ? 'btn-primary'
                          : 'btn-outline-secondary'
                      }`}
                      onClick={() => demo.implemented && selectDemo(category.id, demo.id)}
                      disabled={!demo.implemented}
                    >
                      <div className="d-flex justify-content-between align-items-start">
                        <div className="flex-grow-1">
                          <div className="small fw-semibold">{demo.name}</div>
                          {demo.implemented && (
                            <div className="small text-muted" style={{ fontSize: '0.75rem' }}>
                              {demo.description}
                            </div>
                          )}
                        </div>
                        <div className="d-flex gap-1 ms-2">
                          {demo.implemented && (
                            <span className="badge bg-success" title="Implemented">
                              <Check size={10} />
                            </span>
                          )}
                          {demo.gpuAccelerated && (
                            <span className="badge bg-warning" title="GPU Accelerated">
                              <Zap size={10} />
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="p-3 border-top border-dark bg-dark">
        <div className="small text-muted">
          <div className="d-flex align-items-center mb-1">
            <Check size={12} className="me-1" />
            <span>Implemented</span>
          </div>
          <div className="d-flex align-items-center">
            <Zap size={12} className="me-1" />
            <span>GPU Accelerated</span>
          </div>
        </div>
      </div>
    </div>
  );
}
