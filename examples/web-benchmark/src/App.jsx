import { useEffect } from 'react';
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

import { useAppStore } from './store/appStore';
import { imageToImageData, matToImageDataURL, createThumbnail } from './utils/imageUtils';
import { getDemoById } from './demos/demoRegistry';

import Sidebar from './components/Sidebar';
import DemoControls from './components/DemoControls';
import InputOutput from './components/InputOutput';
import PerformanceMetrics from './components/PerformanceMetrics';
import History from './components/History';

function App() {
  const {
    wasmLoaded,
    gpuAvailable,
    setWasmLoaded,
    setGpuAvailable,
    selectedDemo,
    demoParams,
    inputImage,
    setOutputImage,
    setProcessing,
    setPerformance,
    addToHistory
  } = useAppStore();

  // Initialize WASM and WebGPU once
  useEffect(() => {
    let initialized = false;

    const initWasm = async () => {
      if (initialized) return;
      initialized = true;

      try {
        console.log('Initializing WASM module...');
        await init();

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

  const processImage = async () => {
    if (!inputImage || !wasmLoaded) {
      alert('Please upload an image first');
      return;
    }

    const demo = getDemoById(selectedDemo);
    if (!demo || !demo.implemented) {
      alert('This demo is not yet implemented');
      return;
    }

    setProcessing(true);
    let cpuTime = null;
    let gpuTime = null;
    let resultImage = null;

    try {
      // Load image to ImageData
      const imageData = await imageToImageData(inputImage.dataURL);

      // Create WASM Mat
      const srcMat = WasmMat.fromImageData(
        imageData.data,
        imageData.width,
        imageData.height,
        4
      );

      // Process with GPU (if available)
      if (gpuAvailable) {
        const startGpu = performance.now();
        const gpuResult = await runDemo(demo.id, srcMat, demoParams);
        const endGpu = performance.now();
        gpuTime = endGpu - startGpu;

        if (gpuResult) {
          resultImage = matToImageDataURL(gpuResult);
          gpuResult.free();
        }
      }

      // Process with CPU for comparison
      const startCpu = performance.now();
      const cpuResult = await runDemo(demo.id, srcMat, demoParams);
      const endCpu = performance.now();
      cpuTime = endCpu - startCpu;

      if (cpuResult && !resultImage) {
        resultImage = matToImageDataURL(cpuResult);
      }
      if (cpuResult) {
        cpuResult.free();
      }

      // Clean up source mat
      srcMat.free();

      // Update UI
      if (resultImage) {
        setOutputImage(resultImage);
        setPerformance(cpuTime, gpuTime);

        // Add to history
        const thumbnail = await createThumbnail(resultImage);
        addToHistory({
          category: demo.category,
          demo: demo.id,
          params: { ...demoParams },
          inputImage: inputImage.dataURL,
          outputImage: resultImage,
          outputThumbnail: thumbnail,
          processingTime: gpuTime || cpuTime
        });
      }
    } catch (error) {
      console.error('Processing failed:', error);
      alert(`Processing failed: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const runDemo = async (demoId, srcMat, params) => {
    switch (demoId) {
      case 'gaussian_blur': {
        const ksize = params.ksize || 5;
        const sigma = params.sigma || 1.5;
        return await wasmGaussianBlur(srcMat, ksize, sigma);
      }

      case 'resize': {
        const scale = params.scale || 0.5;
        const newWidth = Math.floor(srcMat.width * scale);
        const newHeight = Math.floor(srcMat.height * scale);
        return await wasmResize(srcMat, newWidth, newHeight);
      }

      case 'threshold': {
        const thresh = params.thresh || 127;
        const maxval = params.maxval || 255;
        return await wasmThreshold(srcMat, thresh, maxval);
      }

      case 'canny': {
        const threshold1 = params.threshold1 || 50;
        const threshold2 = params.threshold2 || 150;
        return await wasmCanny(srcMat, threshold1, threshold2);
      }

      default:
        throw new Error(`Unknown demo: ${demoId}`);
    }
  };

  return (
    <div className="d-flex flex-column vh-100 bg-dark text-light">
      {/* Header */}
      <header className="bg-secondary border-bottom border-primary py-3">
        <div className="container-fluid">
          <div className="d-flex justify-content-between align-items-center">
            <h1 className="h3 mb-0 text-primary">OpenCV-Rust Interactive Demos</h1>
            <div className="d-flex gap-2">
              <span className={`badge ${wasmLoaded ? 'bg-success' : 'bg-warning'}`}>
                {wasmLoaded ? '✓ WASM Ready' : '⏳ Loading...'}
              </span>
              <span className={`badge ${gpuAvailable ? 'bg-success' : 'bg-warning'}`}>
                {gpuAvailable === null ? '⏳ GPU Init...' :
                 gpuAvailable ? '✓ WebGPU' : '⚠ CPU Only'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container-fluid flex-grow-1 overflow-hidden">
        <div className="row h-100">
          {/* Sidebar */}
          <div className="col-md-3 col-lg-2 bg-secondary border-end border-dark p-0 overflow-auto">
            <Sidebar />
          </div>

          {/* Main Panel */}
          <main className="col-md-9 col-lg-10 p-4 overflow-auto">
            <div className="mb-4">
              <DemoControls />
            </div>

            <div className="mb-4">
              <InputOutput onProcess={processImage} />
            </div>

            <div className="mb-4">
              <PerformanceMetrics />
            </div>

            <div>
              <History />
            </div>
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-secondary border-top border-dark py-2 text-center">
        <small className="text-muted">
          OpenCV-Rust v{wasmLoaded && getVersion()} • Pure Rust Image Processing •
          {gpuAvailable ? ' WebGPU Accelerated' : ' CPU Mode'}
        </small>
      </footer>
    </div>
  );
}

export default App;
