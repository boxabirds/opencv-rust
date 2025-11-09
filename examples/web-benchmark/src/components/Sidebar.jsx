import { useState } from 'react';
import { ChevronDown, ChevronRight, Check, Zap } from 'lucide-react';
import { categories, getDemosByCategory } from '../demos/demoRegistry';
import { useAppStore } from '../store/appStore';

export default function Sidebar() {
  const [expandedCategories, setExpandedCategories] = useState(['filters', 'edges', 'transform', 'color']);
  const { selectedCategory, selectedDemo, selectDemo } = useAppStore();

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev =>
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>OpenCV Demos</h2>
        <p className="sidebar-subtitle">Interactive Image Processing</p>
      </div>

      <div className="categories-list">
        {categories.map(category => {
          const demos = getDemosByCategory(category.id);
          const isExpanded = expandedCategories.includes(category.id);
          const implementedCount = demos.filter(d => d.implemented).length;
          const totalCount = demos.length;

          return (
            <div key={category.id} className="category-item">
              <button
                className="category-header"
                onClick={() => toggleCategory(category.id)}
              >
                <span className="category-icon">
                  {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </span>
                <span className="category-name">{category.name}</span>
                <span className="category-count">
                  {implementedCount}/{totalCount}
                </span>
              </button>

              {isExpanded && (
                <div className="demos-list">
                  {demos.map(demo => (
                    <button
                      key={demo.id}
                      className={`demo-item ${
                        selectedCategory === category.id && selectedDemo === demo.id
                          ? 'active'
                          : ''
                      } ${!demo.implemented ? 'disabled' : ''}`}
                      onClick={() => demo.implemented && selectDemo(category.id, demo.id)}
                      disabled={!demo.implemented}
                    >
                      <div className="demo-item-content">
                        <span className="demo-name">{demo.name}</span>
                        <div className="demo-badges">
                          {demo.implemented && (
                            <span className="badge implemented">
                              <Check size={12} />
                            </span>
                          )}
                          {demo.gpuAccelerated && (
                            <span className="badge gpu">
                              <Zap size={12} />
                            </span>
                          )}
                        </div>
                      </div>
                      {demo.implemented && (
                        <p className="demo-description">{demo.description}</p>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="sidebar-footer">
        <div className="legend">
          <div className="legend-item">
            <Check size={14} />
            <span>Implemented</span>
          </div>
          <div className="legend-item">
            <Zap size={14} />
            <span>GPU Accelerated</span>
          </div>
        </div>
      </div>
    </div>
  );
}
