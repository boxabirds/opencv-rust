import { useRef } from 'react';
import { Upload, Play, Camera } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { imageToImageData } from '../utils/imageUtils';

export default function InputOutput({ onProcess }) {
  const fileInputRef = useRef(null);
  const {
    inputImage,
    outputImage,
    setInputImage,
    isProcessing
  } = useAppStore();

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
    <div className="card bg-dark text-light">
      <div className="card-header bg-secondary">
        <div className="d-flex justify-content-between align-items-center">
          <h3 className="h5 mb-0">Input & Output</h3>
          <div className="d-flex gap-2">
            <button
              className="btn btn-secondary"
              onClick={() => fileInputRef.current.click()}
              disabled={isProcessing}
            >
              <Upload size={16} />
              <span className="ms-1">Upload Image</span>
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
              <span className="ms-1">{isProcessing ? 'Processing...' : 'Process Image'}</span>
            </button>
          </div>
        </div>
      </div>

      <div className="card-body">
        <div className="row g-3">
          <div className="col-md-6">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h4 className="h6 mb-0">Input</h4>
              {inputImage && (
                <span className="badge bg-secondary">
                  {inputImage.width} × {inputImage.height}
                </span>
              )}
            </div>
            <div className="border border-secondary rounded bg-dark p-3" style={{ minHeight: '300px' }}>
              {inputImage ? (
                <img src={inputImage.dataURL} alt="Input" className="img-fluid rounded" />
              ) : (
                <div className="d-flex flex-column align-items-center justify-content-center text-light" style={{ minHeight: '260px' }}>
                  <Upload size={48} className="mb-3 text-muted" />
                  <p className="mb-3">Upload an image to begin</p>
                  <button
                    className="btn btn-link text-primary"
                    onClick={() => fileInputRef.current.click()}
                  >
                    Choose File
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="col-md-6">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h4 className="h6 mb-0">Output</h4>
              {outputImage && inputImage && (
                <span className="badge bg-secondary">
                  {inputImage.width} × {inputImage.height}
                </span>
              )}
            </div>
            <div className="border border-secondary rounded bg-dark p-3" style={{ minHeight: '300px' }}>
              {outputImage ? (
                <img src={outputImage} alt="Output" className="img-fluid rounded" />
              ) : (
                <div className="d-flex flex-column align-items-center justify-content-center text-light" style={{ minHeight: '260px' }}>
                  <Play size={48} className="mb-3 text-muted" />
                  <p className="mb-2">Process an image to see results</p>
                  <p className="text-muted small mb-0">
                    {!inputImage
                      ? 'Upload an image first'
                      : 'Click "Process Image" to run'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
