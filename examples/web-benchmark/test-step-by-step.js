#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: 'new', dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

// Step 1: Just check WebGPU initializes
console.log('\n=== STEP 1: WebGPU initialization ===');
try {
  const result = await page.evaluate(async () => {
    if (!navigator.gpu) return 'NO_GPU';
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 'NO_ADAPTER';
    const device = await adapter.requestDevice();
    return device ? 'OK' : 'NO_DEVICE';
  });
  console.log(`✓ Step 1 result: ${result}`);
} catch (e) {
  console.error(`✗ Step 1 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 2: Load WASM module
console.log('\n=== STEP 2: Load WASM ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    await module.default();
    return 'OK';
  });
  console.log(`✓ Step 2 result: ${result}`);
} catch (e) {
  console.error(`✗ Step 2 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 3: Create a WasmMat
console.log('\n=== STEP 3: Create WasmMat ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    const width = 10, height = 10, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);
    mat.free();
    return 'OK';
  });
  console.log(`✓ Step 3 result: ${result}`);
} catch (e) {
  console.error(`✗ Step 3 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 4: Call gaussianBlur with minimal params
console.log('\n=== STEP 4: Call gaussianBlur ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    const width = 10, height = 10, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);

    const blurred = await module.gaussianBlur(mat, 3, 1.0);

    if (blurred) blurred.free();
    mat.free();
    return 'OK';
  });
  console.log(`✓ Step 4 result: ${result}`);
} catch (e) {
  console.error(`✗ Step 4 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

console.log('\n✓ All steps passed!');
await browser.close();
