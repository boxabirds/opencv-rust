import { useState, useEffect } from 'react';
import './App.css';
import init, {
  WasmMat,
  initGpu,
  isGpuAvailable,
  gaussianBlur as wasmGaussianBlur,
  resize as wasmResize,
  threshold as wasmThreshold,
  canny as wasmCanny,
  getVersion
} from '../../../pkg/opencv_rust.js';

function App() {
  const [gpuAvailable, setGpuAvailable] = useState(null);
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState('gaussian_blur');
  const [selectedImage, setSelectedImage] = useState(null);
  const [benchmarkResults, setBenchmarkResults] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [outputImage, setOutputImage] = useState(null);

  // Initialize WASM module
  useEffect(() => {
    const initWasm = async () => {
      try {
        console.log('Initializing WASM module...');
        await init();

        // Check and initialize WebGPU
        console.log('Initializing WebGPU...');
        const gpuInit = await initGpu();
        setGpuAvailable(gpuInit);

        if (gpuInit) {
          console.log('✓ WebGPU initialized successfully');
        } else {
          console.warn('WebGPU initialization failed - falling back to CPU');
        }

        const version = getVersion();
        console.log(`OpenCV-Rust WASM loaded! Version: ${version}`);
        setWasmLoaded(true);
      } catch (error) {
        console.error('Failed to initialize WASM:', error);
        setWasmLoaded(false);
      }
    };

    initWasm();
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

  // Helper function to load image to canvas and get ImageData
  const imageToImageData = async (imageSrc) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = imageSrc;
    });
  };

  // Helper function to convert WasmMat to image data URL for display
  const matToImageDataURL = (mat) => {
    try {
      const width = mat.width();
      const height = mat.height();
      const channels = mat.channels();
      const data = mat.get_data();

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      const imageData = ctx.createImageData(width, height);
      const pixels = imageData.data;

      // Convert Mat data to RGBA
      if (channels === 1) {
        // Grayscale - replicate to RGB
        for (let i = 0; i < data.length; i++) {
          const idx = i * 4;
          pixels[idx] = data[i];     // R
          pixels[idx + 1] = data[i]; // G
          pixels[idx + 2] = data[i]; // B
          pixels[idx + 3] = 255;     // A
        }
      } else if (channels === 3) {
        // BGR to RGBA
        for (let i = 0; i < data.length; i += 3) {
          const idx = (i / 3) * 4;
          pixels[idx] = data[i + 2];     // R (from B)
          pixels[idx + 1] = data[i + 1]; // G
          pixels[idx + 2] = data[i];     // B (from R)
          pixels[idx + 3] = 255;         // A
        }
      } else if (channels === 4) {
        // RGBA - direct copy
        pixels.set(data);
      }

      ctx.putImageData(imageData, 0, 0);
      return canvas.toDataURL();
    } catch (error) {
      console.error('Failed to convert Mat to image:', error);
      return null;
    }
  };

  const runBenchmark = async (mode) => {
    if (!selectedImage) {
      alert('Please upload an image first');
      return;
    }

    if (!wasmLoaded) {
      alert('WASM module not loaded yet. Please wait...');
      return;
    }

    setIsRunning(true);

    try {
      // Load image to ImageData
      const imageData = await imageToImageData(selectedImage.data);

      // Create WASM Mat from ImageData
      const srcMat = WasmMat.fromImageData(
        imageData.data,
        imageData.width,
        imageData.height,
        4  // RGBA
      );

      const iterations = 10;
      const start = performance.now();
      let finalResult = null;

      for (let i = 0; i < iterations; i++) {
        let result;

        switch (selectedOperation) {
          case 'gaussian_blur':
            result = await wasmGaussianBlur(srcMat, 5, 1.5);
            break;
          case 'resize':
            const newWidth = Math.floor(imageData.width * 0.5);
            const newHeight = Math.floor(imageData.height * 0.5);
            result = await wasmResize(srcMat, newWidth, newHeight);
            break;
          case 'threshold':
            result = await wasmThreshold(srcMat, 128, 255);
            break;
          case 'canny':
            result = await wasmCanny(srcMat, 50, 150);
            break;
          default:
            throw new Error(`Unknown operation: ${selectedOperation}`);
        }

        // Keep the last result for display, free previous results
        if (i === iterations - 1) {
          finalResult = result;
        } else if (result) {
          result.free();
        }
      }

      const end = performance.now();
      const avgTime = (end - start) / iterations;

      // Convert result to image for display
      if (finalResult) {
        const outputDataURL = matToImageDataURL(finalResult);
        setOutputImage(outputDataURL);
        finalResult.free();
      }

      // Clean up source mat
      srcMat.free();

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
          <span className={`status-badge ${wasmLoaded ? 'success' : 'warning'}`}>
            {wasmLoaded ? '✓ WASM Loaded' : '⚠ WASM Loading...'}
          </span>
          <span className={`status-badge ${gpuAvailable ? 'success' : 'error'}`}>
            {gpuAvailable === null ? '⏳ Initializing WebGPU...' :
             gpuAvailable ? '✓ WebGPU Ready' : '✗ WebGPU Failed (CPU fallback)'}
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
            <h2>Input & Output</h2>
            <div className="image-comparison">
              <div className="image-container">
                <h3>Input Image</h3>
                <div className="image-preview">
                  {selectedImage ? (
                    <img src={selectedImage.data} alt="Input" />
                  ) : (
                    <div className="placeholder">
                      <p>📷 Upload an image to begin</p>
                    </div>
                  )}
                </div>
              </div>
              <div className="image-container">
                <h3>Output Image</h3>
                <div className="image-preview">
                  {outputImage ? (
                    <img src={outputImage} alt="Output" />
                  ) : (
                    <div className="placeholder">
                      <p>🎯 Run benchmark to see results</p>
                    </div>
                  )}
                </div>
              </div>
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
          <strong>WebGPU Acceleration:</strong> This demo uses WebGPU for GPU-accelerated image processing.
          All operations run on your graphics card for maximum performance.
        </p>
        <p>
          <strong>Browser Requirements:</strong> Chrome/Edge 113+ or Firefox Nightly with WebGPU enabled.
          Enable at <code>chrome://flags/#enable-unsafe-webgpu</code> if needed.
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
