import { Zap, Cpu } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { formatTime } from '../utils/imageUtils';

export default function PerformanceMetrics() {
  const { lastCpuTime, lastGpuTime, lastSpeedup, gpuAvailable } = useAppStore();

  if (!lastCpuTime && !lastGpuTime) {
    return (
      <div className="performance-section">
        <h3>Performance Metrics</h3>
        <div className="performance-placeholder">
          <p>Run an operation to see performance metrics</p>
        </div>
      </div>
    );
  }

  const hasGpu = gpuAvailable && lastGpuTime;

  return (
    <div className="performance-section">
      <h3>Performance Metrics</h3>
      <div className="performance-grid">
        {lastCpuTime && (
          <div className="performance-card cpu">
            <div className="card-icon">
              <Cpu size={24} />
            </div>
            <div className="card-content">
              <div className="card-label">CPU Time</div>
              <div className="card-value">{formatTime(lastCpuTime)}</div>
            </div>
          </div>
        )}

        {hasGpu && (
          <div className="performance-card gpu">
            <div className="card-icon">
              <Zap size={24} />
            </div>
            <div className="card-content">
              <div className="card-label">GPU Time</div>
              <div className="card-value">{formatTime(lastGpuTime)}</div>
            </div>
          </div>
        )}

        {hasGpu && lastSpeedup && (
          <div className="performance-card speedup">
            <div className="card-icon">
              <span className="speedup-icon">⚡</span>
            </div>
            <div className="card-content">
              <div className="card-label">Speedup</div>
              <div className="card-value">{lastSpeedup}x</div>
              <div className="card-hint">
                {parseFloat(lastSpeedup) > 10 ? 'Excellent!' :
                 parseFloat(lastSpeedup) > 5 ? 'Very Good' :
                 parseFloat(lastSpeedup) > 2 ? 'Good' : 'Modest'}
              </div>
            </div>
          </div>
        )}
      </div>

      {!gpuAvailable && (
        <div className="performance-note">
          <p>
            <strong>Note:</strong> WebGPU is not available. All operations run on CPU.
            For GPU acceleration, use Chrome/Edge 113+ or enable WebGPU in your browser.
          </p>
        </div>
      )}
    </div>
  );
}
