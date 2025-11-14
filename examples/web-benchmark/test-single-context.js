#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

console.log('\n=== Testing Gaussian Blur in single context ===');
try {
  const result = await page.evaluate(async () => {
    // Wait for the app to fully initialize (same as web-benchmark)
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Use the already-loaded and initialized WASM module from window
    if (!window.opencv_rust) {
      return 'WASM_NOT_LOADED';
    }

    console.log('Creating test image...');
    const width = 1024, height = 1024, channels = 4;
    const data = new Uint8ClampedArray(width * height * channels).fill(128);

    // Use WasmMat from the global scope (same as web-benchmark)
    const WasmMat = window.opencv_rust.WasmMat;
    const gaussianBlur = window.opencv_rust.gaussianBlur;

    console.log('Creating WasmMat...');
    const mat = WasmMat.fromImageData(data, width, height, channels);

    console.log('Running Gaussian blur...');
    const blurred = await gaussianBlur(mat, 5, 1.5);

    if (blurred) blurred.free();
    mat.free();
    return 'OK';
  });

  console.log(`✓ Result: ${result}`);
} catch (e) {
  console.error(`✗ CRASH: ${e.message}`);
}

await browser.close();
