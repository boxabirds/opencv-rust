import { create } from 'zustand';

export const useAppStore = create((set, get) => ({
  // WASM state
  wasmLoaded: false,
  gpuAvailable: null,

  // Current demo state
  selectedCategory: 'filters',
  selectedDemo: 'gaussian_blur',
  demoParams: {},

  // Input/Output state
  inputImage: null,
  outputImage: null,
  isProcessing: false,

  // Performance tracking
  lastCpuTime: null,
  lastGpuTime: null,
  lastSpeedup: null,

  // History (last 20 results)
  history: [],
  maxHistory: 20,

  // Animation state
  isAnimating: false,
  animationProgress: 0,
  animationTotal: 0,

  // Actions
  setWasmLoaded: (loaded) => set({ wasmLoaded: loaded }),
  setGpuAvailable: (available) => set({ gpuAvailable: available }),

  selectDemo: (category, demo) => set({
    selectedCategory: category,
    selectedDemo: demo,
    demoParams: {} // Reset params when switching demos
  }),

  setDemoParam: (key, value) => set((state) => ({
    demoParams: { ...state.demoParams, [key]: value }
  })),

  setInputImage: (image) => set({ inputImage: image }),

  setOutputImage: (image) => set({ outputImage: image }),

  setProcessing: (isProcessing) => set({ isProcessing }),

  setPerformance: (cpuTime, gpuTime) => {
    const speedup = cpuTime && gpuTime ? (cpuTime / gpuTime).toFixed(2) : null;
    set({
      lastCpuTime: cpuTime,
      lastGpuTime: gpuTime,
      lastSpeedup: speedup
    });
  },

  addToHistory: (result) => set((state) => {
    const newHistory = [
      {
        ...result,
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString()
      },
      ...state.history
    ].slice(0, state.maxHistory);

    return { history: newHistory };
  }),

  restoreFromHistory: (historyItem) => set({
    selectedCategory: historyItem.category,
    selectedDemo: historyItem.demo,
    demoParams: historyItem.params,
    inputImage: historyItem.inputImage,
    outputImage: historyItem.outputImage
  }),

  clearHistory: () => set({ history: [] }),

  setAnimating: (isAnimating) => set({ isAnimating }),

  setAnimationProgress: (progress, total) => set({
    animationProgress: progress,
    animationTotal: total
  }),
}));
