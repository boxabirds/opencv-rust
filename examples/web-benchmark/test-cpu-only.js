#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: 'new' });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

console.log('\n Testing Gaussian Blur with CPU backend...');

try {
  const result = await page.evaluate(async () => {
    const module = await import('./pkg/opencv_rust.js');
    await module.default();

    // Force CPU backend
    window.backend = 'cpu';

    const width = 50, height = 50, channels = 4;
    const data = new Uint8Array(width * height * channels).fill(128);
    const mat = module.WasmMat.fromImageData(data, width, height, channels);

    const result = await module.gaussianBlur(mat, 5, 1.0);
    if (result) result.free();
    mat.free();

    return 'OK';
  });

  console.log(`✓ CPU backend: ${result}`);
} catch (e) {
  console.error(`✗ CPU backend CRASH: ${e.message}`);
}

await browser.close();
