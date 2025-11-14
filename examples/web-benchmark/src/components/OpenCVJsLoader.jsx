/**
 * OpenCV.js Dynamic Loader (Web Worker version)
 *
 * Loads opencv.js in a Web Worker to prevent main thread blocking/crashes.
 * Provides helper functions to run operations for performance benchmarking.
 */

import { useEffect, useState } from 'react';

// Global state
let worker = null;
let opencvLoaded = false;
let opencvLoading = false;
let messageId = 0;
const pendingMessages = new Map();

/**
 * Create and initialize the Web Worker
 */
const createWorker = () => {
  if (worker) {
    return worker;
  }

  console.log('[Main] Creating OpenCV.js Web Worker...');
  worker = new Worker('/opencv-worker.js');

  worker.onmessage = (event) => {
    const { type, id, result, error } = event.data;

    const pending = pendingMessages.get(id);
    if (!pending) {
      console.warn('[Main] Received message for unknown ID:', id);
      return;
    }

    pendingMessages.delete(id);

    switch (type) {
      case 'init_success':
        console.log('[Main] ✓ OpenCV.js Web Worker initialized');
        opencvLoaded = true;
        opencvLoading = false;
        pending.resolve();
        break;

      case 'operation_result':
      case 'benchmark_result':
        pending.resolve(result);
        break;

      case 'error':
        console.error('[Main] Worker error:', error);
        pending.reject(new Error(error));
        break;

      default:
        console.warn('[Main] Unknown message type:', type);
    }
  };

  worker.onerror = (error) => {
    console.error('[Main] Worker error:', error);
    opencvLoading = false;

    // Reject all pending messages
    for (const [id, pending] of pendingMessages.entries()) {
      pending.reject(error);
    }
    pendingMessages.clear();
  };

  return worker;
};

/**
 * Send a message to the worker and wait for response
 */
const sendWorkerMessage = (message) => {
  return new Promise((resolve, reject) => {
    const id = messageId++;
    const w = createWorker();

    // Set timeout to prevent indefinite hang
    const timeout = setTimeout(() => {
      pendingMessages.delete(id);
      reject(new Error(`Worker message timeout (${message.type})`));
    }, 60000); // 60s timeout

    pendingMessages.set(id, {
      resolve: (result) => {
        clearTimeout(timeout);
        resolve(result);
      },
      reject: (error) => {
        clearTimeout(timeout);
        reject(error);
      }
    });

    w.postMessage({ ...message, id });
  });
};

/**
 * Load OpenCV.js in the Web Worker
 */
export const loadOpenCVJs = () => {
  if (opencvLoaded) {
    return Promise.resolve();
  }

  if (opencvLoading) {
    // Wait for current load to complete
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        if (opencvLoaded) {
          clearInterval(checkInterval);
          resolve();
        } else if (!opencvLoading) {
          clearInterval(checkInterval);
          reject(new Error('OpenCV.js loading failed'));
        }
      }, 100);

      // Timeout after 60s
      setTimeout(() => {
        clearInterval(checkInterval);
        if (!opencvLoaded) {
          reject(new Error('OpenCV.js loading timeout'));
        }
      }, 60000);
    });
  }

  opencvLoading = true;
  console.log('[Main] Loading OpenCV.js (9.5MB) in Web Worker - this prevents tab freezing');

  return sendWorkerMessage({ type: 'init' });
};

/**
 * Check if OpenCV.js is loaded
 */
export const isOpenCVJsLoaded = () => {
  return opencvLoaded;
};

/**
 * React hook to load OpenCV.js (lazy loading - only when needed)
 * @param {boolean} shouldLoad - Set to true to trigger loading
 */
export const useOpenCVJs = (shouldLoad = false) => {
  const [loaded, setLoaded] = useState(opencvLoaded);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Don't auto-load unless explicitly requested
    if (!shouldLoad) {
      return;
    }

    if (opencvLoaded) {
      setLoaded(true);
      return;
    }

    setLoading(true);
    loadOpenCVJs()
      .then(() => {
        setLoaded(true);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        setLoading(false);
      });
  }, [shouldLoad]);

  return { loaded, loading, error };
};

/**
 * Run an OpenCV.js operation in the Web Worker
 */
export const runOpenCVJsOperation = async (operationName, imageData, params) => {
  if (!opencvLoaded) {
    throw new Error('OpenCV.js not loaded');
  }

  try {
    const result = await sendWorkerMessage({
      type: 'run_operation',
      operation: operationName,
      imageData: {
        data: imageData.data,
        width: imageData.width,
        height: imageData.height
      },
      params
    });

    // Convert back to ImageData object
    return new ImageData(
      new Uint8ClampedArray(result.data),
      result.width,
      result.height
    );
  } catch (error) {
    console.error(`OpenCV.js operation ${operationName} failed:`, error);
    throw error;
  }
};

/**
 * Benchmark an operation with opencv.js in the Web Worker
 */
export const benchmarkOpenCVJs = async (operationName, imageData, params, iterations = 10) => {
  if (!opencvLoaded) {
    throw new Error('OpenCV.js not loaded');
  }

  try {
    return await sendWorkerMessage({
      type: 'benchmark',
      operation: operationName,
      imageData: {
        data: imageData.data,
        width: imageData.width,
        height: imageData.height
      },
      params: { ...params, iterations }
    });
  } catch (error) {
    console.error(`OpenCV.js benchmark ${operationName} failed:`, error);
    throw error;
  }
};

/**
 * Cleanup worker (call on app unmount if needed)
 */
export const terminateWorker = () => {
  if (worker) {
    worker.terminate();
    worker = null;
    opencvLoaded = false;
    opencvLoading = false;
    pendingMessages.clear();
    console.log('[Main] OpenCV.js Web Worker terminated');
  }
};

export default {
  loadOpenCVJs,
  isOpenCVJsLoaded,
  useOpenCVJs,
  runOpenCVJsOperation,
  benchmarkOpenCVJs,
  terminateWorker,
};
