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
      <div className="demo-controls">
        <p className="no-params">No parameters for this operation</p>
      </div>
    );
  }

  const renderParam = (param) => {
    const value = demoParams[param.id] ?? param.default;

    switch (param.type) {
      case 'slider':
        return (
          <div key={param.id} className="control-group">
            <label>
              <span className="control-label">{param.name}</span>
              <span className="control-value">{value}</span>
            </label>
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={value}
              onChange={(e) => setDemoParam(param.id, parseFloat(e.target.value))}
              className="slider"
            />
            {param.description && (
              <p className="control-description">{param.description}</p>
            )}
          </div>
        );

      case 'select':
        return (
          <div key={param.id} className="control-group">
            <label>
              <span className="control-label">{param.name}</span>
            </label>
            <select
              value={value}
              onChange={(e) => setDemoParam(param.id, e.target.value)}
              className="select"
            >
              {param.options.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {param.description && (
              <p className="control-description">{param.description}</p>
            )}
          </div>
        );

      case 'color':
        return (
          <div key={param.id} className="control-group">
            <label>
              <span className="control-label">{param.name}</span>
              <span className="control-value">{value}</span>
            </label>
            <input
              type="color"
              value={value}
              onChange={(e) => setDemoParam(param.id, e.target.value)}
              className="color-picker"
            />
            {param.description && (
              <p className="control-description">{param.description}</p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="demo-controls">
      <div className="demo-header">
        <h3>{demo.name}</h3>
        <p className="demo-description">{demo.description}</p>
      </div>
      <div className="controls-grid">
        {demo.params.map(renderParam)}
      </div>
    </div>
  );
}
