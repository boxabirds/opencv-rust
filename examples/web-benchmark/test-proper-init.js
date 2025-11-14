#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

console.log('\n=== Step 1: Initialize WASM once at page level ===');
await page.evaluate(async () => {
  // Import and initialize WASM module at page scope
  const module = await import('./pkg/opencv_rust.js');
  await module.default();

  // Expose to window so we can use it in subsequent evaluate() calls
  window.opencv_rust = module;
  console.log('WASM module loaded and exposed to window');
});
console.log('✓ WASM initialized');

console.log('\n=== Step 2: Create WasmMat using initialized module ===');
const matCreated = await page.evaluate(() => {
  const { WasmMat } = window.opencv_rust;
  const width = 1024, height = 1024, channels = 4;
  const data = new Uint8ClampedArray(width * height * channels).fill(128);
  window.testMat = WasmMat.fromImageData(data, width, height, channels);
  console.log(`Created mat: ${window.testMat.rows()}x${window.testMat.cols()}`);
  return 'OK';
});
console.log(`✓ Mat created: ${matCreated}`);

console.log('\n=== Step 3: Call gaussianBlur on the mat ===');
try {
  const result = await page.evaluate(async () => {
    const { gaussianBlur } = window.opencv_rust;
    console.log('Calling gaussianBlur with ksize=5, sigma=1.5...');
    const blurred = await gaussianBlur(window.testMat, 5, 1.5);
    console.log(`Result: ${blurred.rows()}x${blurred.cols()}`);
    blurred.free();
    return 'OK';
  });
  console.log(`✓ Gaussian blur succeeded: ${result}`);
} catch (e) {
  console.error(`✗ Gaussian blur CRASH: ${e.message}`);
}

console.log('\n=== Step 4: Call it again (should work since shaders compiled) ===');
try {
  const result = await page.evaluate(async () => {
    const { gaussianBlur } = window.opencv_rust;
    console.log('Calling gaussianBlur second time...');
    const blurred = await gaussianBlur(window.testMat, 5, 1.5);
    console.log(`Result: ${blurred.rows()}x${blurred.cols()}`);
    blurred.free();
    return 'OK';
  });
  console.log(`✓ Second call succeeded: ${result}`);
} catch (e) {
  console.error(`✗ Second call CRASH: ${e.message}`);
}

console.log('\n=== Step 5: Cleanup ===');
await page.evaluate(() => {
  window.testMat.free();
  console.log('Mat freed');
});

console.log('\n✓ All tests passed!');
await browser.close();
