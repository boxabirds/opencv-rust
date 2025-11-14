#!/usr/bin/env node
import puppeteer from 'puppeteer';

const tests = [
  { name: 'Gaussian Blur', id: 'gaussian_blur', params: [5, 1.0] },
  { name: 'Box Blur', id: 'box_blur', params: [5] },
  { name: 'Median Blur', id: 'median_blur', params: [5] },
  { name: 'Bilateral Filter', id: 'bilateral_filter', params: [9, 75.0, 75.0] },
  { name: 'Guided Filter', id: 'guided_filter', params: [5, 0.1] },
  { name: 'Gabor Filter', id: 'gabor_filter', params: [21, 3.0, 0, 5.0, 1.0, 0] },
  { name: 'LoG', id: 'log_filter', params: [5, 1.0] },
  { name: 'NLM Denoising', id: 'nlm_denoising', params: [10, 7, 21] },
  { name: 'Anisotropic Diffusion', id: 'anisotropic_diffusion', params: [5, 10.0, 0.25] },
  { name: 'Distance Transform', id: 'distance_transform', params: [] },
];

const browser = await puppeteer.launch({ headless: 'new', dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

for (let i = 0; i < tests.length; i++) {
  const test = tests[i];
  console.log(`\n[${i+1}/${tests.length}] Testing ${test.name}...`);

  try {
    const result = await page.evaluate(async (testId, params) => {
      const module = await import('./pkg/opencv_rust.js');
      if (!window.wasmInitialized) {
        await module.default();
        window.wasmInitialized = true;
      }

      const width = 50, height = 50, channels = 4;
      const data = new Uint8Array(width * height * channels).fill(128);
      const mat = module.WasmMat.fromImageData(data, width, height, channels);

      const functionName = testId.split('_')
        .map((word, idx) => idx === 0 ? word : word.charAt(0).toUpperCase() + word.slice(1))
        .join('');

      if (!module[functionName]) {
        throw new Error('NOT_EXPORTED');
      }

      const result = await module[functionName](mat, ...params);
      if (result) result.free();
      mat.free();

      return 'OK';
    }, test.id, test.params);

    console.log(`✓ ${result}`);
  } catch (e) {
    console.error(`✗ CRASH: ${e.message}`);
    console.error(`\n*** CRASHED ON TEST #${i+1}: ${test.name} (${test.id}) ***\n`);
    await browser.close();
    process.exit(1);
  }
}

console.log('\n✓ All tests passed!');
await browser.close();
