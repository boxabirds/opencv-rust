import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [gpuAvailable, setGpuAvailable] = useState(null);
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState('gaussian_blur');
  const [selectedImage, setSelectedImage] = useState(null);
  const [benchmarkResults, setBenchmarkResults] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  // Check WebGPU availability
  useEffect(() => {
    const checkGPU = async () => {
      if (!navigator.gpu) {
        setGpuAvailable(false);
        return;
      }

      try {
        const adapter = await navigator.gpu.requestAdapter();
        setGpuAvailable(!!adapter);
      } catch (error) {
        console.error('GPU check failed:', error);
        setGpuAvailable(false);
      }
    };

    checkGPU();
  }, []);

  const operations = [
    { id: 'gaussian_blur', name: 'Gaussian Blur', description: 'Smooth images with Gaussian filter' },
    { id: 'resize', name: 'Resize', description: 'Scale images up or down' },
    { id: 'threshold', name: 'Threshold', description: 'Binary threshold operations' },
    { id: 'canny', name: 'Canny Edge', description: 'Detect edges in images' },
  ];

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setSelectedImage({
            file,
            data: event.target.result,
            width: img.width,
            height: img.height,
          });
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const runBenchmark = async (mode) => {
    if (!selectedImage) {
      alert('Please upload an image first');
      return;
    }

    setIsRunning(true);

    try {
      // Simulate benchmark (replace with actual WASM calls)
      const iterations = 10;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        // TODO: Call WASM function
        // await opencv_rust.gaussian_blur(imageData);
        await new Promise(resolve => setTimeout(resolve, 10)); // Simulate work
      }

      const end = performance.now();
      const avgTime = (end - start) / iterations;

      setBenchmarkResults(prev => [
        ...prev,
        {
          operation: selectedOperation,
          mode,
          time: avgTime.toFixed(2),
          throughput: (1000 / avgTime).toFixed(2),
          imageSize: `${selectedImage.width}x${selectedImage.height}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } catch (error) {
      console.error('Benchmark failed:', error);
      alert(`Benchmark failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🚀 OpenCV-Rust WebGPU Benchmark</h1>
        <div className="status">
          <span className={`status-badge ${gpuAvailable ? 'success' : 'error'}`}>
            {gpuAvailable === null ? '⏳ Checking GPU...' :
             gpuAvailable ? '✓ WebGPU Available' : '✗ WebGPU Not Available'}
          </span>
          <span className={`status-badge ${wasmLoaded ? 'success' : 'warning'}`}>
            {wasmLoaded ? '✓ WASM Loaded' : '⚠ WASM Not Loaded'}
          </span>
        </div>
      </header>

      <div className="container">
        <div className="sidebar">
          <section className="card">
            <h2>1. Select Operation</h2>
            <div className="operation-list">
              {operations.map(op => (
                <button
                  key={op.id}
                  className={`operation-btn ${selectedOperation === op.id ? 'active' : ''}`}
                  onClick={() => setSelectedOperation(op.id)}
                >
                  <div className="operation-name">{op.name}</div>
                  <div className="operation-desc">{op.description}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="card">
            <h2>2. Upload Image</h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="file-input"
            />
            {selectedImage && (
              <div className="image-info">
                <p><strong>Size:</strong> {selectedImage.width}x{selectedImage.height}</p>
                <p><strong>File:</strong> {selectedImage.file.name}</p>
              </div>
            )}
          </section>

          <section className="card">
            <h2>3. Run Benchmark</h2>
            <button
              className="bench-btn cpu"
              onClick={() => runBenchmark('CPU')}
              disabled={isRunning || !selectedImage}
            >
              {isRunning ? '⏳ Running...' : '▶ Run CPU (WASM)'}
            </button>
            <button
              className="bench-btn gpu"
              onClick={() => runBenchmark('GPU')}
              disabled={isRunning || !selectedImage || !gpuAvailable}
            >
              {isRunning ? '⏳ Running...' : '▶ Run GPU (WebGPU)'}
            </button>
            <button
              className="bench-btn both"
              onClick={async () => {
                await runBenchmark('CPU');
                await runBenchmark('GPU');
              }}
              disabled={isRunning || !selectedImage || !gpuAvailable}
            >
              {isRunning ? '⏳ Running...' : '▶ Run Both'}
            </button>
          </section>
        </div>

        <div className="main-content">
          <section className="card preview-section">
            <h2>Image Preview</h2>
            <div className="image-preview">
              {selectedImage ? (
                <img src={selectedImage.data} alt="Uploaded" />
              ) : (
                <div className="placeholder">
                  <p>📷 Upload an image to begin</p>
                </div>
              )}
            </div>
          </section>

          <section className="card results-section">
            <h2>Benchmark Results</h2>
            {benchmarkResults.length === 0 ? (
              <div className="placeholder">
                <p>No results yet. Run a benchmark to see performance metrics.</p>
              </div>
            ) : (
              <div className="results-table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>Operation</th>
                      <th>Mode</th>
                      <th>Image Size</th>
                      <th>Time (ms)</th>
                      <th>Throughput (ops/sec)</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResults.map((result, idx) => (
                      <tr key={idx} className={result.mode.toLowerCase()}>
                        <td>{operations.find(op => op.id === result.operation)?.name}</td>
                        <td>
                          <span className={`mode-badge ${result.mode.toLowerCase()}`}>
                            {result.mode}
                          </span>
                        </td>
                        <td>{result.imageSize}</td>
                        <td>{result.time}</td>
                        <td>{result.throughput}</td>
                        <td>{result.timestamp}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>

      <footer className="footer">
        <p>
          <strong>Note:</strong> This demo requires WebGPU support.
          Enable it in Chrome/Edge at <code>chrome://flags/#enable-unsafe-webgpu</code>
        </p>
        <p>
          <a href="https://github.com/boxabirds/opencv-rust" target="_blank" rel="noopener noreferrer">
            📖 View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
