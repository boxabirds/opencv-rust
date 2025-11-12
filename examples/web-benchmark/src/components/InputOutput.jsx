import { useRef, useEffect, useState } from 'react';
import { Upload, Play, Camera } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { imageToImageData } from '../utils/imageUtils';
import BenchmarkComparison from './BenchmarkComparison';

export default function InputOutput({ onProcess }) {
  const fileInputRef = useRef(null);
  const {
    inputImage,
    outputImage,
    setInputImage,
    isProcessing,
    selectedDemo,
    demoParams
  } = useAppStore();

  // Debug: Log when outputImage changes
  useEffect(() => {
    console.log('[InputOutput] outputImage changed:', outputImage ? `data URL (${outputImage.length} chars)` : 'NULL');
  }, [outputImage]);

  const [inputImageData, setInputImageData] = useState(null);

  // Convert input image to ImageData for comparison
  useEffect(() => {
    if (inputImage && inputImage.dataURL) {
      imageToImageData(inputImage.dataURL)
        .then(data => setInputImageData(data))
        .catch(err => console.error('Failed to convert image:', err));
    } else {
      setInputImageData(null);
    }
  }, [inputImage]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setInputImage({
            file,
            dataURL: event.target.result,
            width: img.width,
            height: img.height,
          });
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="input-output-section">
      <div className="section-header">
        <h3>Input & Output</h3>
        <div className="action-buttons">
          <button
            className="btn btn-secondary"
            onClick={() => fileInputRef.current.click()}
            disabled={isProcessing}
          >
            <Upload size={16} />
            Upload Image
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            className="btn btn-primary"
            onClick={onProcess}
            disabled={!inputImage || isProcessing}
          >
            <Play size={16} />
            {isProcessing ? 'Processing...' : 'Process Image'}
          </button>
        </div>
      </div>

      <div className="image-comparison">
        <div className="image-panel">
          <div className="panel-header">
            <h4>Input</h4>
            {inputImage && (
              <span className="image-info">
                {inputImage.width} × {inputImage.height}
              </span>
            )}
          </div>
          <div className="image-container">
            {inputImage ? (
              <img src={inputImage.dataURL} alt="Input" />
            ) : (
              <div className="placeholder">
                <Upload size={48} />
                <p>Upload an image to begin</p>
                <button
                  className="btn btn-link"
                  onClick={() => fileInputRef.current.click()}
                >
                  Choose File
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="image-panel">
          <div className="panel-header">
            <h4>Output</h4>
            {outputImage && inputImage && (
              <span className="image-info">
                {inputImage.width} × {inputImage.height}
              </span>
            )}
          </div>
          <div className="image-container">
            {outputImage ? (
              <>
                <img
                  src={outputImage}
                  alt="Output"
                  onLoad={() => console.log('[InputOutput] Output image loaded successfully')}
                  onError={(e) => console.error('[InputOutput] Output image failed to load:', e)}
                />
              </>
            ) : (
              <div className="placeholder">
                <Play size={48} />
                <p>Process an image to see results</p>
                <p className="placeholder-hint">
                  {!inputImage
                    ? 'Upload an image first'
                    : 'Click "Process Image" to run'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Benchmark Comparison with OpenCV.js */}
      {inputImageData && outputImage && (
        <BenchmarkComparison
          operationName={selectedDemo}
          imageData={inputImageData}
          params={demoParams}
        />
      )}
    </div>
  );
}
