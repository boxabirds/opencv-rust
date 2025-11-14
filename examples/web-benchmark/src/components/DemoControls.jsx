import { useAppStore } from '../store/appStore';
import { getDemoById, getDefaultParams } from '../demos/demoRegistry';
import { useEffect, useRef } from 'react';
import {
  generateParameterCombinations,
  createLoopingSequence,
  estimateAnimationDuration
} from '../utils/animationUtils';

export default function DemoControls({ onProcess }) {
  const {
    selectedDemo,
    demoParams,
    setDemoParam,
    isAnimating,
    animationProgress,
    animationTotal,
    setAnimating,
    setAnimationProgress,
    gpuAvailable,
    inputImage
  } = useAppStore();
  const demo = getDemoById(selectedDemo);
  const animationRef = useRef(null);

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

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        if (animationRef.current.timeout) {
          clearTimeout(animationRef.current.timeout);
        }
        setAnimating(false);
      }
    };
  }, []);

  const startAnimation = async () => {
    if (!demo || !onProcess || !inputImage) {
      alert('Please upload an image first');
      return;
    }

    if (!gpuAvailable) {
      alert('Animation requires GPU acceleration (WebGPU)');
      return;
    }

    // Generate all parameter combinations
    const combinations = generateParameterCombinations(demo);
    const sequence = createLoopingSequence(combinations);

    console.log(`Starting animation with ${sequence.length} frames`);
    console.log('Warming up GPU pipelines (3 warmup runs)...');

    setAnimating(true);
    setAnimationProgress(0, sequence.length);

    // Store the active flag in ref immediately
    animationRef.current = { active: true, timeout: null };

    // Warmup phase - compile and cache GPU pipelines
    // Run first 3 unique parameter combinations to ensure pipelines are compiled
    const warmupCombinations = combinations.slice(0, Math.min(3, combinations.length));
    for (let i = 0; i < warmupCombinations.length; i++) {
      // Check if stopped during warmup
      if (!animationRef.current || !animationRef.current.active) {
        console.log('Animation stopped during warmup');
        return;
      }

      const params = warmupCombinations[i];
      Object.entries(params).forEach(([key, value]) => {
        setDemoParam(key, value);
      });
      try {
        console.log(`Warmup ${i + 1}/${warmupCombinations.length}...`);
        await onProcess();
      } catch (error) {
        console.error('Warmup error:', error);
      }
    }

    console.log('✓ Warmup complete, starting animation loop');

    let currentFrame = 0;

    const animateFrame = async () => {
      // Check if animation was stopped
      if (!animationRef.current || !animationRef.current.active) {
        console.log('Animation stopped');
        return;
      }

      const state = useAppStore.getState();
      if (!state.isAnimating) {
        console.log('Animation stopped (state check)');
        return;
      }

      if (currentFrame >= sequence.length) {
        // Animation complete - loop back to start
        currentFrame = 0;
      }

      // Update parameters
      const params = sequence[currentFrame];
      Object.entries(params).forEach(([key, value]) => {
        setDemoParam(key, value);
      });

      // Process with current params
      try {
        await onProcess();
      } catch (error) {
        console.error('Animation frame error:', error);
        if (animationRef.current) {
          animationRef.current.active = false;
        }
        stopAnimation();
        return;
      }

      setAnimationProgress(currentFrame + 1, sequence.length);
      currentFrame++;

      // Schedule next frame only if still animating
      if (animationRef.current && animationRef.current.active && state.isAnimating) {
        animationRef.current.timeout = setTimeout(animateFrame, 200); // 200ms delay between frames
      }
    };

    // Start the animation
    animateFrame();
  };

  const stopAnimation = () => {
    console.log('Stopping animation...');
    setAnimating(false);
    setAnimationProgress(0, 0);

    if (animationRef.current) {
      if (animationRef.current.timeout) {
        clearTimeout(animationRef.current.timeout);
      }
      if (animationRef.current.active !== undefined) {
        animationRef.current.active = false;
      }
      animationRef.current = null;
    }
  };

  const handleAnimateClick = () => {
    if (isAnimating) {
      stopAnimation();
    } else {
      startAnimation();
    }
  };

  if (!demo) {
    return (
      <div className="demo-controls">
        <p className="no-params">No demo selected</p>
      </div>
    );
  }

  const hasParams = demo.params && demo.params.length > 0;
  const animationInfo = hasParams ? estimateAnimationDuration(demo) : null;

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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ margin: 0 }}>{demo.name}</h3>
            <p className="demo-description" style={{ margin: '0.5rem 0 0 0' }}>{demo.description}</p>
          </div>
          {hasParams && (
            <button
              onClick={handleAnimateClick}
              disabled={!gpuAvailable || !inputImage}
              className={`btn ${isAnimating ? 'btn-secondary' : 'btn-primary'}`}
              style={{ whiteSpace: 'nowrap' }}
              title={animationInfo ? `${animationInfo.frames} frames (~${animationInfo.durationSec}s per loop)` : ''}
            >
              {isAnimating ? (
                <>
                  <span>⏹</span>
                  <span>Stop ({animationProgress}/{animationTotal})</span>
                </>
              ) : (
                <>
                  <span>▶</span>
                  <span>Animate</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
      {hasParams ? (
        <div className="controls-grid">
          {demo.params.map(renderParam)}
        </div>
      ) : (
        <p className="no-params">No parameters for this operation</p>
      )}
    </div>
  );
}
