#!/usr/bin/env node
/**
 * Automated test suite for ALL WASM endpoints
 * Actually calls each function with test data and reports results
 */

import init, { WasmMat, initGpu } from '../../pkg/opencv_rust.js';
import { demos } from './src/demos/demoRegistry.js';
import fs from 'fs';

// Create test images of different sizes/types
function createTestImage(width = 640, height = 480, channels = 4) {
  const size = width * height * channels;
  const data = new Uint8Array(size);

  // Create a gradient pattern
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * channels;
      data[idx + 0] = (x / width) * 255;     // R
      data[idx + 1] = (y / height) * 255;    // G
      data[idx + 2] = 128;                    // B
      if (channels === 4) {
        data[idx + 3] = 255;                  // A
      }
    }
  }

  return { data, width, height, channels };
}

// Test a single operation
async function testOperation(demo, testImages) {
  const result = {
    id: demo.id,
    name: demo.name,
    category: demo.category,
    status: 'UNKNOWN',
    error: null,
    duration: 0
  };

  try {
    // Get the WASM function
    const module = await import('../../pkg/opencv_rust.js');
    const functionName = demo.id.split('_')
      .map((word, idx) => idx === 0 ? word : word.charAt(0).toUpperCase() + word.slice(1))
      .join('');

    const wasmFunc = module[functionName];

    if (!wasmFunc) {
      result.status = 'NOT_EXPORTED';
      result.error = `Function '${functionName}' not found in WASM module`;
      return result;
    }

    // Create WASM Mat from test image
    const testImg = testImages.rgba;
    const srcMat = WasmMat.fromImageData(
      testImg.data,
      testImg.width,
      testImg.height,
      testImg.channels
    );

    // Get default parameters
    const params = {};
    if (demo.params) {
      demo.params.forEach(param => {
        params[param.id] = param.default;
      });
    }

    // Build argument list
    const args = [srcMat];
    if (demo.params) {
      demo.params.forEach(param => {
        args.push(params[param.id]);
      });
    }

    // Call the function
    const start = Date.now();
    const output = await wasmFunc(...args);
    const duration = Date.now() - start;

    // Validate output
    if (!output) {
      result.status = 'NULL_OUTPUT';
      result.error = 'Function returned null/undefined';
      srcMat.free();
      return result;
    }

    // Check if output is valid
    if (typeof output.width !== 'number' || typeof output.height !== 'number') {
      result.status = 'INVALID_OUTPUT';
      result.error = 'Output missing width/height properties';
      srcMat.free();
      if (output.free) output.free();
      return result;
    }

    // Success!
    result.status = 'OK';
    result.duration = duration;
    srcMat.free();
    if (output.free) output.free();

  } catch (error) {
    result.status = 'ERROR';
    result.error = error.message || error.toString();

    // Categorize error types
    if (result.error.includes('Source must have 3 channels')) {
      result.errorType = 'CHANNEL_MISMATCH';
    } else if (result.error.includes('unreachable')) {
      result.errorType = 'UNIMPLEMENTED';
    } else if (result.error.includes('RuntimeError')) {
      result.errorType = 'WASM_PANIC';
    } else if (result.error.includes('Invalid parameter')) {
      result.errorType = 'INVALID_PARAM';
    } else {
      result.errorType = 'UNKNOWN_ERROR';
    }
  }

  return result;
}

async function main() {
  console.log('='.repeat(80));
  console.log('AUTOMATED WASM ENDPOINT TEST SUITE');
  console.log('='.repeat(80));
  console.log('');

  // Initialize
  console.log('⏳ Initializing WASM module...');
  await init();
  console.log('✓ WASM initialized');

  console.log('⏳ Initializing GPU...');
  try {
    await initGpu();
    console.log('✓ GPU initialized');
  } catch (e) {
    console.log('⚠ GPU initialization failed, will test CPU fallback');
  }

  // Create test images
  console.log('⏳ Creating test images...');
  const testImages = {
    rgba: createTestImage(640, 480, 4),
    rgb: createTestImage(640, 480, 3),
    small: createTestImage(320, 240, 4),
    large: createTestImage(1024, 1024, 4)
  };
  console.log('✓ Test images created');
  console.log('');

  // Run tests
  console.log('='.repeat(80));
  console.log(`TESTING ${demos.length} OPERATIONS`);
  console.log('='.repeat(80));
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
    unknownError: []
  };

  let completed = 0;
  for (const demo of demos) {
    completed++;
    process.stdout.write(`[${completed}/${demos.length}] Testing ${demo.name}...`);

    const result = await testOperation(demo, testImages);

    switch (result.status) {
      case 'OK':
        results.ok.push(result);
        console.log(` ✓ ${result.duration}ms`);
        break;
      case 'NOT_EXPORTED':
        results.notExported.push(result);
        console.log(` ✗ NOT_EXPORTED`);
        break;
      case 'NULL_OUTPUT':
        results.nullOutput.push(result);
        console.log(` ✗ NULL_OUTPUT`);
        break;
      case 'INVALID_OUTPUT':
        results.invalidOutput.push(result);
        console.log(` ✗ INVALID_OUTPUT`);
        break;
      case 'ERROR':
        switch (result.errorType) {
          case 'CHANNEL_MISMATCH':
            results.channelMismatch.push(result);
            console.log(` ✗ CHANNEL_MISMATCH`);
            break;
          case 'UNIMPLEMENTED':
            results.unimplemented.push(result);
            console.log(` ✗ UNIMPLEMENTED`);
            break;
          case 'WASM_PANIC':
            results.wasmPanic.push(result);
            console.log(` ✗ WASM_PANIC`);
            break;
          case 'INVALID_PARAM':
            results.invalidParam.push(result);
            console.log(` ✗ INVALID_PARAM`);
            break;
          default:
            results.unknownError.push(result);
            console.log(` ✗ ERROR`);
        }
        break;
    }
  }

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
  console.log(`  ✗ Null output:      ${results.nullOutput.length}`);
  console.log(`  ✗ Invalid output:   ${results.invalidOutput.length}`);
  console.log(`  ✗ Channel mismatch: ${results.channelMismatch.length}`);
  console.log(`  ✗ Unimplemented:    ${results.unimplemented.length}`);
  console.log(`  ✗ WASM panic:       ${results.wasmPanic.length}`);
  console.log(`  ✗ Invalid param:    ${results.invalidParam.length}`);
  console.log(`  ✗ Unknown error:    ${results.unknownError.length}`);

  // Detailed failure reports
  const failureCategories = [
    { name: 'CHANNEL MISMATCH', items: results.channelMismatch },
    { name: 'NOT EXPORTED', items: results.notExported },
    { name: 'UNIMPLEMENTED', items: results.unimplemented },
    { name: 'WASM PANIC', items: results.wasmPanic },
    { name: 'INVALID PARAMETER', items: results.invalidParam },
    { name: 'NULL OUTPUT', items: results.nullOutput },
    { name: 'INVALID OUTPUT', items: results.invalidOutput },
    { name: 'UNKNOWN ERROR', items: results.unknownError }
  ];

  for (const category of failureCategories) {
    if (category.items.length > 0) {
      console.log('');
      console.log('='.repeat(80));
      console.log(`${category.name} (${category.items.length})`);
      console.log('='.repeat(80));

      // Group by category
      const byCategory = {};
      for (const item of category.items) {
        if (!byCategory[item.category]) {
          byCategory[item.category] = [];
        }
        byCategory[item.category].push(item);
      }

      for (const [cat, items] of Object.entries(byCategory)) {
        console.log(`\n${cat}:`);
        for (const item of items) {
          console.log(`  - ${item.name} (${item.id})`);
          if (item.error) {
            console.log(`    Error: ${item.error.substring(0, 100)}`);
          }
        }
      }
    }
  }

  // Performance stats for passing tests
  if (results.ok.length > 0) {
    console.log('');
    console.log('='.repeat(80));
    console.log('PERFORMANCE (passing tests only)');
    console.log('='.repeat(80));

    const durations = results.ok.map(r => r.duration);
    const avg = durations.reduce((a, b) => a + b, 0) / durations.length;
    const min = Math.min(...durations);
    const max = Math.max(...durations);
    const sorted = [...durations].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];

    console.log(`Average: ${avg.toFixed(2)}ms`);
    console.log(`Median:  ${median.toFixed(2)}ms`);
    console.log(`Min:     ${min.toFixed(2)}ms`);
    console.log(`Max:     ${max.toFixed(2)}ms`);
  }

  // Save detailed results
  const outputFile = 'wasm-test-results.json';
  fs.writeFileSync(outputFile, JSON.stringify({
    timestamp: new Date().toISOString(),
    summary: {
      total,
      passing,
      failing,
      passRate
    },
    results
  }, null, 2));

  console.log('');
  console.log('='.repeat(80));
  console.log(`✓ Detailed results saved to ${outputFile}`);
  console.log('='.repeat(80));

  // Exit with error if any tests failed
  process.exit(failing > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
