#!/usr/bin/env node
/**
 * Run WASM tests in isolation - each test gets a fresh page load
 * This prevents one crashing test from stopping the entire suite
 */

import puppeteer from 'puppeteer';
import fs from 'fs';

const TEST_URL = 'http://localhost:3000/test-suite.html';
const OUTPUT_FILE = 'wasm-test-results.json';
const PER_TEST_TIMEOUT_MS = 30000; // 30 seconds per test

async function testSingleOperation(browser, demoId, demoName, totalTests, currentIndex) {
  const page = await browser.newPage();

  try {
    // Load the test page
    await page.goto(TEST_URL, { waitUntil: 'networkidle0', timeout: 15000 });

    // Wait for initialization
    await page.waitForFunction(() => {
      const btn = document.getElementById('startBtn');
      return btn && !btn.disabled;
    }, { timeout: 15000 });

    // Test just this one operation
    const result = await page.evaluate(async (id) => {
      const { demos } = await import('./src/demos/demoRegistry.js');
      const demo = demos.find(d => d.id === id);
      if (!demo) {
        return { status: 'NOT_FOUND', error: 'Demo not found in registry' };
      }

      // Create test image
      const createTestImage = (width = 640, height = 480, channels = 4) => {
        const size = width * height * channels;
        const data = new Uint8Array(size);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * channels;
            data[idx + 0] = (x / width) * 255;
            data[idx + 1] = (y / height) * 255;
            data[idx + 2] = 128;
            if (channels === 4) data[idx + 3] = 255;
          }
        }
        return { data, width, height, channels };
      };

      const testImg = createTestImage();
      const { WasmMat } = await import('../../pkg/opencv_rust.js');
      const srcMat = WasmMat.fromImageData(testImg.data, testImg.width, testImg.height, testImg.channels);

      // Get params
      const params = {};
      if (demo.params) {
        demo.params.forEach(param => {
          params[param.id] = param.default;
        });
      }

      // Call via the global function we'll inject
      try {
        const start = performance.now();
        const output = await window.callDemoOperation(await import('../../pkg/opencv_rust.js'), demo.id, srcMat, params);
        const duration = performance.now() - start;

        if (!output || typeof output.width !== 'number') {
          return { status: 'INVALID_OUTPUT', error: 'Invalid output' };
        }

        srcMat.free();
        if (output.free) output.free();

        return { status: 'OK', duration };
      } catch (error) {
        const errorMsg = error.message || error.toString();

        if (errorMsg.includes('not found in WASM module')) {
          return { status: 'NOT_EXPORTED', error: errorMsg };
        } else if (errorMsg.includes('Source must have 3 channels')) {
          return { status: 'ERROR', errorType: 'CHANNEL_MISMATCH', error: errorMsg };
        } else if (errorMsg.includes('unreachable')) {
          return { status: 'ERROR', errorType: 'UNIMPLEMENTED', error: errorMsg };
        } else if (errorMsg.includes('RuntimeError') || errorMsg.includes('index out of bounds')) {
          return { status: 'ERROR', errorType: 'WASM_PANIC', error: errorMsg };
        } else if (errorMsg.includes('Invalid parameter')) {
          return { status: 'ERROR', errorType: 'INVALID_PARAM', error: errorMsg };
        } else {
          return { status: 'ERROR', errorType: 'UNKNOWN_ERROR', error: errorMsg };
        }
      }
    }, demoId);

    console.log(`[${currentIndex}/${totalTests}] ${demoName}: ${result.status}${result.duration ? ` (${result.duration.toFixed(2)}ms)` : ''}`);

    await page.close();
    return { ...result, id: demoId, name: demoName };

  } catch (error) {
    console.log(`[${currentIndex}/${totalTests}] ${demoName}: CRASH (${error.message})`);
    await page.close().catch(() => {});
    return {
      id: demoId,
      name: demoName,
      status: 'ERROR',
      errorType: 'PAGE_CRASH',
      error: error.message
    };
  }
}

async function main() {
  console.log('='.repeat(80));
  console.log('ISOLATED WASM ENDPOINT TEST SUITE');
  console.log('Each test runs in a fresh page to isolate crashes');
  console.log('='.repeat(80));
  console.log('');

  let browser;
  try {
    console.log('🚀 Launching browser...');
    browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--no-sandbox',
        '--disable-setuid-sandbox'
      ]
    });

    // Get the list of demos
    console.log('📋 Loading demo list...');
    const demosModule = await import('./src/demos/demoRegistry.js');
    const demos = demosModule.demos;
    console.log(`Found ${demos.length} operations to test`);
    console.log('');

    const results = {
      ok: [],
      notExported: [],
      nullOutput: [],
      invalidOutput: [],
      channelMismatch: [],
      unimplemented: [],
      wasmPanic: [],
      invalidParam: [],
      unknownError: [],
      pageCrash: []
    };

    // Test each operation in isolation
    for (let i = 0; i < demos.length; i++) {
      const demo = demos[i];
      const result = await testSingleOperation(browser, demo.id, demo.name, demos.length, i + 1);

      // Categorize result
      switch (result.status) {
        case 'OK':
          results.ok.push(result);
          break;
        case 'NOT_EXPORTED':
          results.notExported.push(result);
          break;
        case 'INVALID_OUTPUT':
          results.invalidOutput.push(result);
          break;
        case 'ERROR':
          switch (result.errorType) {
            case 'CHANNEL_MISMATCH':
              results.channelMismatch.push(result);
              break;
            case 'UNIMPLEMENTED':
              results.unimplemented.push(result);
              break;
            case 'WASM_PANIC':
              results.wasmPanic.push(result);
              break;
            case 'INVALID_PARAM':
              results.invalidParam.push(result);
              break;
            case 'PAGE_CRASH':
              results.pageCrash.push(result);
              break;
            default:
              results.unknownError.push(result);
          }
          break;
      }

      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    await browser.close();

    // Print summary
    console.log('');
    console.log('='.repeat(80));
    console.log('SUMMARY');
    console.log('='.repeat(80));
    const total = demos.length;
    const passing = results.ok.length;
    const failing = total - passing;
    const passRate = ((passing / total) * 100).toFixed(1);

    console.log(`Total operations:     ${total}`);
    console.log(`✓ Passing:            ${passing} (${passRate}%)`);
    console.log(`✗ Failing:            ${failing} (${(100 - passRate).toFixed(1)}%)`);
    console.log('');
    console.log('Breakdown:');
    console.log(`  ✓ OK:               ${results.ok.length}`);
    console.log(`  ✗ Not exported:     ${results.notExported.length}`);
    console.log(`  ✗ Invalid output:   ${results.invalidOutput.length}`);
    console.log(`  ✗ Channel mismatch: ${results.channelMismatch.length}`);
    console.log(`  ✗ Unimplemented:    ${results.unimplemented.length}`);
    console.log(`  ✗ WASM panic:       ${results.wasmPanic.length}`);
    console.log(`  ✗ Invalid param:    ${results.invalidParam.length}`);
    console.log(`  ✗ Page crash:       ${results.pageCrash.length}`);
    console.log(`  ✗ Unknown error:    ${results.unknownError.length}`);

    // Save results
    const output = {
      timestamp: new Date().toISOString(),
      summary: { total, passing, failing, passRate },
      results
    };

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
    console.log('');
    console.log(`✓ Results saved to ${OUTPUT_FILE}`);
    console.log('='.repeat(80));

    // Print failures
    const failureCategories = [
      { name: 'CHANNEL MISMATCH', items: results.channelMismatch },
      { name: 'NOT EXPORTED', items: results.notExported },
      { name: 'UNIMPLEMENTED', items: results.unimplemented },
      { name: 'WASM PANIC', items: results.wasmPanic },
      { name: 'PAGE CRASH', items: results.pageCrash },
      { name: 'INVALID PARAMETER', items: results.invalidParam },
      { name: 'INVALID OUTPUT', items: results.invalidOutput },
      { name: 'UNKNOWN ERROR', items: results.unknownError }
    ];

    for (const category of failureCategories) {
      if (category.items.length > 0) {
        console.log('');
        console.log('='.repeat(80));
        console.log(`${category.name} (${category.items.length})`);
        console.log('='.repeat(80));
        for (const item of category.items) {
          console.log(`  - ${item.name} (${item.id})`);
          if (item.error) {
            const errorPreview = item.error.substring(0, 80);
            console.log(`    ${errorPreview}${item.error.length > 80 ? '...' : ''}`);
          }
        }
      }
    }

    process.exit(failing > 0 ? 1 : 0);

  } catch (error) {
    console.error('');
    console.error('='.repeat(80));
    console.error('FATAL ERROR');
    console.error('='.repeat(80));
    console.error(error.message);
    console.error(error.stack);

    if (browser) {
      await browser.close();
    }

    process.exit(1);
  }
}

main();
