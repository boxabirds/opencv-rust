#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

// Step 1: Initialize WebGPU manually
console.log('\n=== STEP 1: Manual WebGPU init ===');
try {
  const result = await page.evaluate(async () => {
    if (!navigator.gpu) return 'NO_GPU';
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 'NO_ADAPTER';
    window.testAdapter = adapter;
    window.testDevice = await adapter.requestDevice();
    return 'OK';
  });
  console.log(`✓ Step 1: ${result}`);
} catch (e) {
  console.error(`✗ Step 1 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 2: Create a simple buffer
console.log('\n=== STEP 2: Create simple GPU buffer ===');
try {
  const result = await page.evaluate(async () => {
    const device = window.testDevice;
    const buffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false,
    });
    return 'OK';
  });
  console.log(`✓ Step 2: ${result}`);
} catch (e) {
  console.error(`✗ Step 2 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 3: Load WASM and get GPU context from it
console.log('\n=== STEP 3: Load WASM and check GPU context ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    await module.default();
    return 'OK';
  });
  console.log(`✓ Step 3: ${result}`);
} catch (e) {
  console.error(`✗ Step 3 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 4: Call WasmMat.fromImageData (no operations yet)
console.log('\n=== STEP 4: Create WasmMat only ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    const width = 10, height = 10, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);
    mat.free();
    return 'OK';
  });
  console.log(`✓ Step 4: ${result}`);
} catch (e) {
  console.error(`✗ Step 4 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 5: Try calling boxBlur (simpler operation)
console.log('\n=== STEP 5: Try boxBlur instead ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    const width = 10, height = 10, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);

    const blurred = await module.boxBlur(mat, 3);

    if (blurred) blurred.free();
    mat.free();
    return 'OK';
  });
  console.log(`✓ Step 5: ${result}`);
} catch (e) {
  console.error(`✗ Step 5 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

// Step 6: Now try gaussianBlur WITH WARMUP (like web-benchmark does)
console.log('\n=== STEP 6: Try gaussianBlur (with warmup) ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    const width = 1024, height = 1024, channels = 4;
    const data = new Uint8ClampedArray(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);

    // WARMUP RUN (compile GPU shaders)
    console.log('Running warmup to compile shaders...');
    const warmup = await module.gaussianBlur(mat, 5, 1.5);
    if (warmup) warmup.free();

    // ACTUAL RUN
    console.log('Running actual test...');
    const blurred = await module.gaussianBlur(mat, 5, 1.5);

    if (blurred) blurred.free();
    mat.free();
    return 'OK';
  });
  console.log(`✓ Step 6: ${result}`);
} catch (e) {
  console.error(`✗ Step 6 CRASH: ${e.message}`);
  await browser.close();
  process.exit(1);
}

console.log('\n✓ All steps passed!');
await browser.close();
