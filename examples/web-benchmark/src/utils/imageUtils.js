/**
 * Utility functions for image processing and conversion
 */

/**
 * Load image from URL/data URL to ImageData
 */
export const imageToImageData = async (imageSrc) => {
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

/**
 * Convert WasmMat to image data URL for display
 */
export const matToImageDataURL = (mat) => {
  try {
    // Access as properties, not functions!
    const width = mat.width;
    const height = mat.height;
    const channels = mat.channels;
    const data = mat.getData();

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

/**
 * Create a thumbnail from an image data URL
 */
export const createThumbnail = (imageDataURL, maxWidth = 150, maxHeight = 150) => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      let width = img.width;
      let height = img.height;

      // Calculate new dimensions maintaining aspect ratio
      if (width > height) {
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
      } else {
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL());
    };
    img.src = imageDataURL;
  });
};

/**
 * Format time in ms to readable string
 */
export const formatTime = (ms) => {
  if (ms === null || ms === undefined) return 'N/A';
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

/**
 * Calculate speedup ratio
 */
export const calculateSpeedup = (cpuTime, gpuTime) => {
  if (!cpuTime || !gpuTime) return null;
  return (cpuTime / gpuTime).toFixed(2);
};
