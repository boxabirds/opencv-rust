#!/usr/bin/env node
import puppeteer from 'puppeteer';
import { demos } from './src/demos/demoRegistry.js';

const implemented = demos.filter(d => d.implemented);

async function testIncrementally() {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  page.on('console', msg => console.log('PAGE:', msg.text()));
  page.on('pageerror', err => console.error('ERROR:', err.message));

  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

  // Test each operation one by one
  for (let i = 0; i < implemented.length; i++) {
    const demo = implemented[i];
    console.log(`\n[${i+1}/${implemented.length}] Testing ${demo.name} (${demo.id})...`);

    const result = await page.evaluate(async (demoId) => {
      try {
        const module = await import('./pkg/opencv_rust.js');
        await module.default();

        const width = 100, height = 100, channels = 4;
        const data = new Uint8Array(width * height * channels).fill(128);
        const mat = module.WasmMat.fromImageData(data, width, height, channels);

        const functionName = demoId.split('_')
          .map((word, idx) => idx === 0 ? word : word.charAt(0).toUpperCase() + word.slice(1))
          .join('');

        if (!module[functionName]) {
          return { success: false, error: 'NOT_EXPORTED' };
        }

        // Call with minimal params
        let resultMat;
        if (demoId === 'gaussian_blur') {
          resultMat = await module[functionName](mat, 5, 1.0);
        } else if (demoId === 'bilateral_filter') {
          resultMat = await module[functionName](mat, 9, 75.0, 75.0);
        } else if (demoId === 'guided_filter') {
          resultMat = await module[functionName](mat, 5, 0.1);
        } else if (demoId === 'gabor_filter') {
          resultMat = await module[functionName](mat, 21, 3.0, 0, 5.0, 1.0, 0);
        } else if (demoId === 'log_filter') {
          resultMat = await module[functionName](mat, 5, 1.0);
        } else if (demoId === 'nlm_denoising') {
          resultMat = await module[functionName](mat, 10, 7, 21);
        } else if (demoId === 'anisotropic_diffusion') {
          resultMat = await module[functionName](mat, 5, 10.0, 0.25);
        } else {
          // Default: call with just the mat
          resultMat = await module[functionName](mat);
        }

        if (resultMat) resultMat.free();
        mat.free();

        return { success: true };
      } catch (e) {
        return { success: false, error: e.message };
      }
    }, demo.id);

    if (!result.success) {
      console.error(`✗ FAILED: ${result.error}`);
      if (result.error !== 'NOT_EXPORTED') {
        console.error(`\n*** CRASH DETECTED AT TEST #${i+1}: ${demo.name} (${demo.id}) ***\n`);
        await browser.close();
        process.exit(1);
      }
    } else {
      console.log(`✓ PASSED`);
    }
  }

  console.log('\n✓ All tests passed!');
  await browser.close();
}

testIncrementally().catch(console.error);
