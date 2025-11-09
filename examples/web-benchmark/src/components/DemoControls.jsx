import { useAppStore } from '../store/appStore';
import { getDemoById, getDefaultParams } from '../demos/demoRegistry';
import { useEffect } from 'react';

export default function DemoControls() {
  const { selectedDemo, demoParams, setDemoParam } = useAppStore();
  const demo = getDemoById(selectedDemo);

  // Initialize params with defaults when demo changes
  useEffect(() => {
    if (demo) {
      const defaults = getDefaultParams(demo.id);
      Object.entries(defaults).forEach(([key, value]) => {
        if (demoParams[key] === undefined) {
          setDemoParam(key, value);
        }
      });
    }
  }, [selectedDemo]);

  if (!demo || demo.params.length === 0) {
    return (
      <div className="card bg-dark text-light">
        <div className="card-body">
          <p className="text-center text-light mb-0">No parameters for this operation</p>
        </div>
      </div>
    );
  }

  const renderParam = (param) => {
    const value = demoParams[param.id] ?? param.default;

    switch (param.type) {
      case 'slider':
        return (
          <div key={param.id} className="mb-3">
            <label className="form-label">
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-light">{param.name}</span>
                <span className="badge bg-secondary">{value}</span>
              </div>
            </label>
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={value}
              onChange={(e) => setDemoParam(param.id, parseFloat(e.target.value))}
              className="form-range"
            />
            {param.description && (
              <p className="text-muted small mt-1 mb-0">{param.description}</p>
            )}
          </div>
        );

      case 'select':
        return (
          <div key={param.id} className="mb-3">
            <label className="form-label text-light">
              {param.name}
            </label>
            <select
              value={value}
              onChange={(e) => setDemoParam(param.id, e.target.value)}
              className="form-select bg-dark text-light border-secondary"
            >
              {param.options.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {param.description && (
              <p className="text-muted small mt-1 mb-0">{param.description}</p>
            )}
          </div>
        );

      case 'color':
        return (
          <div key={param.id} className="mb-3">
            <label className="form-label">
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-light">{param.name}</span>
                <span className="badge bg-secondary">{value}</span>
              </div>
            </label>
            <input
              type="color"
              value={value}
              onChange={(e) => setDemoParam(param.id, e.target.value)}
              className="form-control form-control-color bg-dark border-secondary"
              style={{ width: '100%', height: '40px' }}
            />
            {param.description && (
              <p className="text-muted small mt-1 mb-0">{param.description}</p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="card bg-dark text-light">
      <div className="card-body">
        <div className="mb-4">
          <h3 className="card-title h5">{demo.name}</h3>
          <p className="text-muted mb-0">{demo.description}</p>
        </div>
        <div>
          {demo.params.map(renderParam)}
        </div>
      </div>
    </div>
  );
}
