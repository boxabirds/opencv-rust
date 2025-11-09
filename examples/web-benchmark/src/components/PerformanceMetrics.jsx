import { Zap, Cpu } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { formatTime } from '../utils/imageUtils';

export default function PerformanceMetrics() {
  const { lastCpuTime, lastGpuTime, lastSpeedup, gpuAvailable } = useAppStore();

  if (!lastCpuTime && !lastGpuTime) {
    return (
      <div className="card bg-dark text-light">
        <div className="card-body">
          <h3 className="card-title h5">Performance Metrics</h3>
          <div className="text-center text-light py-4">
            <p>Run an operation to see performance metrics</p>
          </div>
        </div>
      </div>
    );
  }

  const hasGpu = gpuAvailable && lastGpuTime;

  return (
    <div className="card bg-dark text-light">
      <div className="card-body">
        <h3 className="card-title h5 mb-3">Performance Metrics</h3>
        <div className="row g-3">
          {lastCpuTime && (
            <div className="col-md-4">
              <div className="card bg-secondary text-light h-100">
                <div className="card-body d-flex align-items-center">
                  <div className="me-3">
                    <Cpu size={24} />
                  </div>
                  <div className="flex-grow-1">
                    <div className="text-muted small">CPU Time</div>
                    <div className="h4 mb-0">{formatTime(lastCpuTime)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {hasGpu && (
            <div className="col-md-4">
              <div className="card bg-primary text-light h-100">
                <div className="card-body d-flex align-items-center">
                  <div className="me-3">
                    <Zap size={24} />
                  </div>
                  <div className="flex-grow-1">
                    <div className="text-light opacity-75 small">GPU Time</div>
                    <div className="h4 mb-0">{formatTime(lastGpuTime)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {hasGpu && lastSpeedup && (
            <div className="col-md-4">
              <div className="card bg-success text-light h-100">
                <div className="card-body d-flex align-items-center">
                  <div className="me-3 fs-2">
                    ⚡
                  </div>
                  <div className="flex-grow-1">
                    <div className="text-light opacity-75 small">Speedup</div>
                    <div className="h4 mb-0">{lastSpeedup}x</div>
                    <div className="mt-1">
                      <span className="badge bg-light text-dark">
                        {parseFloat(lastSpeedup) > 10 ? 'Excellent!' :
                         parseFloat(lastSpeedup) > 5 ? 'Very Good' :
                         parseFloat(lastSpeedup) > 2 ? 'Good' : 'Modest'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {!gpuAvailable && (
          <div className="alert alert-info mt-3 mb-0">
            <p className="mb-0">
              <strong>Note:</strong> WebGPU is not available. All operations run on CPU.
              For GPU acceleration, use Chrome/Edge 113+ or enable WebGPU in your browser.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
