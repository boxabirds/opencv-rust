#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: 'new', dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

// Test if we can call create_gaussian_kernel via WASM
console.log('\n=== Testing kernel creation ===');
try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    await module.default();

    // Try to call a Rust function that creates Gaussian kernel
    // We'll use gaussianBlur but with CPU backend forced
    window.backend = 'cpu';

    const width = 10, height = 10, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);

    const blurred = await module.gaussianBlur(mat, 3, 1.0);

    if (blurred) blurred.free();
    mat.free();
    return 'OK';
  });
  console.log(`✓ CPU backend: ${result}`);
} catch (e) {
  console.error(`✗ CPU backend CRASH: ${e.message}`);
}

await browser.close();
