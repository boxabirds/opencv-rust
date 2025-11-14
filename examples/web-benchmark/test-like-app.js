#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

console.log('\n=== Wait for app to initialize ===');
await new Promise(resolve => setTimeout(resolve, 2000));

console.log('\n=== Testing Gaussian Blur (same pattern as App.jsx) ===');
try {
  const result = await page.evaluate(async () => {
    // Use the imports from the page scope (loaded by App.jsx during init)
    // This mimics exactly how App.jsx works - no dynamic imports!

    // Create test image (just like imageToImageData returns)
    const width = 1024, height = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'gray';
    ctx.fillRect(0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);

    console.log(`[test] Input image: ${imageData.width}x${imageData.height}, ${imageData.data.length} bytes`);

    // Get WasmMat and gaussianBlur from window (exposed by App.jsx)
    // Actually, they're not on window - they're module-scoped in App.jsx
    // So we need to check what's actually available...

    return 'Need to check what is exposed to window';
  });
  console.log(`Result: ${result}`);
} catch (e) {
  console.error(`✗ CRASH: ${e.message}`);
}

console.log('\n=== Check what is available on window ===');
const windowProps = await page.evaluate(() => {
  return Object.keys(window).filter(k =>
    k.toLowerCase().includes('wasm') ||
    k.toLowerCase().includes('mat') ||
    k.toLowerCase().includes('opencv') ||
    k.toLowerCase().includes('gpu')
  );
});
console.log('Relevant window properties:', windowProps);

await browser.close();
