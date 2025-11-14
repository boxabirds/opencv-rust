#!/usr/bin/env node
/**
 * Unit test for Gabor filter to debug the hang
 */

import init, { WasmMat } from '../../pkg/opencv_rust.js';

async function testGabor() {
  console.log('Initializing WASM...');
  await init();
  console.log('✓ WASM initialized');

  // Create a tiny test image
  const width = 32;
  const height = 32;
  const channels = 4;
  const size = width * height * channels;
  const data = new Uint8Array(size);

  // Fill with gradient
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * channels;
      data[idx + 0] = (x / width) * 255;
      data[idx + 1] = (y / height) * 255;
      data[idx + 2] = 128;
      data[idx + 3] = 255;
    }
  }

  console.log(`Creating WasmMat (${width}x${height}, ${channels} channels)...`);
  const srcMat = WasmMat.fromImageData(data, width, height, channels);
  console.log(`✓ WasmMat created: ${srcMat.width}x${srcMat.height}`);

  // Test Gabor filter
  const ksize = 21;
  const sigma = 3.0;
  const theta = 0;
  const lambda = 1.0 / 0.1;
  const gamma = 0.5;
  const psi = 0;

  console.log('');
  console.log('Testing Gabor filter...');
  console.log(`  ksize: ${ksize}`);
  console.log(`  sigma: ${sigma}`);
  console.log(`  theta: ${theta}`);
  console.log(`  lambda: ${lambda}`);
  console.log(`  gamma: ${gamma}`);
  console.log(`  psi: ${psi}`);
  console.log('');

  console.log('Calling gaborFilter...');
  const start = Date.now();

  // Add timeout
  const timeoutMs = 5000;
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => reject(new Error(`Timeout after ${timeoutMs}ms`)), timeoutMs);
  });

  try {
    const module = await import('../../pkg/opencv_rust.js');
    const resultPromise = module.gaborFilter(srcMat, ksize, sigma, theta, lambda, gamma, psi);

    const result = await Promise.race([resultPromise, timeoutPromise]);
    const duration = Date.now() - start;

    console.log(`✓ Gabor filter completed in ${duration}ms`);
    console.log(`  Result: ${result.width}x${result.height}`);

    srcMat.free();
    result.free();

    process.exit(0);
  } catch (error) {
    const duration = Date.now() - start;
    console.error(`✗ Gabor filter failed after ${duration}ms`);
    console.error(`  Error: ${error.message}`);
    console.error(error.stack);

    srcMat.free();
    process.exit(1);
  }
}

testGabor();
