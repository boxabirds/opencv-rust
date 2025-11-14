/**
 * Test all operations systematically by actually running them
 */

import init, { WasmMat, initGpu } from '../../pkg/opencv_rust.js';
import { demos } from './src/demos/demoRegistry.js';
import fs from 'fs';
import { createCanvas, loadImage } from 'canvas';

// Create a simple test image
function createTestImage() {
  const width = 640;
  const height = 480;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Create a gradient
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#ff0000');
  gradient.addColorStop(0.5, '#00ff00');
  gradient.addColorStop(1, '#0000ff');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  // Add some shapes
  ctx.fillStyle = 'white';
  ctx.fillRect(100, 100, 200, 200);
  ctx.beginPath();
  ctx.arc(400, 240, 100, 0, Math.PI * 2);
  ctx.fill();

  const imageData = ctx.getImageData(0, 0, width, height);
  return imageData;
}

async function testOperation(demo, imageData) {
  try {
    // Create WASM Mat from ImageData
    const mat = WasmMat.fromImageData(
      imageData.data,
      imageData.width,
      imageData.height,
      4
    );

    // Get the operation function dynamically
    const module = await import('../../pkg/opencv_rust.js');

    // Map demo ID to function name (handle special cases)
    const functionName = demo.id
      .split('_')
      .map((word, idx) => idx === 0 ? word : word.charAt(0).toUpperCase() + word.slice(1))
      .join('');

    const operation = module[functionName];

    if (!operation) {
      mat.free();
      return {
        id: demo.id,
        name: demo.name,
        status: 'NOT_FOUND',
        error: `Function '${functionName}' not found in WASM module`
      };
    }

    // Get default params
    const params = {};
    if (demo.params) {
      demo.params.forEach(param => {
        params[param.id] = param.default;
      });
    }

    // Build argument list based on the operation
    let args = [mat];

    // Add params based on demo configuration
    if (demo.params && demo.params.length > 0) {
      demo.params.forEach(param => {
        if (param.type === 'slider' || param.type === 'select') {
          args.push(params[param.id]);
        }
      });
    }

    // Try to run the operation
    try {
      const result = await operation(...args);

      // Verify result
      if (!result) {
        mat.free();
        return {
          id: demo.id,
          name: demo.name,
          status: 'NULL_RESULT',
          error: 'Operation returned null/undefined'
        };
      }

      if (!result.width && !result.cols) {
        mat.free();
        if (result.free) result.free();
        return {
          id: demo.id,
          name: demo.name,
          status: 'INVALID_RESULT',
          error: 'Result has no width/cols property'
        };
      }

      // Success
      mat.free();
      if (result.free) result.free();
      return {
        id: demo.id,
        name: demo.name,
        category: demo.category,
        status: 'OK',
        error: null
      };
    } catch (e) {
      mat.free();
      return {
        id: demo.id,
        name: demo.name,
        category: demo.category,
        status: 'RUNTIME_ERROR',
        error: e.message || e.toString()
      };
    }
  } catch (e) {
    return {
      id: demo.id,
      name: demo.name,
      category: demo.category,
      status: 'TEST_ERROR',
      error: e.message || e.toString()
    };
  }
}

async function main() {
  console.log('Initializing WASM module...');
  await init();

  console.log('Initializing GPU...');
  try {
    await initGpu();
    console.log('✓ GPU initialized');
  } catch (e) {
    console.log('⚠ GPU initialization failed (will test CPU path):', e.message);
  }

  console.log('Creating test image...');
  const imageData = createTestImage();

  console.log(`\nTesting ${demos.length} operations...\n`);

  const results = {
    ok: [],
    notFound: [],
    nullResult: [],
    invalidResult: [],
    runtimeError: [],
    testError: []
  };

  for (let i = 0; i < demos.length; i++) {
    const demo = demos[i];
    process.stdout.write(`[${i + 1}/${demos.length}] Testing ${demo.name}...`);

    const result = await testOperation(demo, imageData);

    switch (result.status) {
      case 'OK':
        results.ok.push(result);
        console.log(' ✓');
        break;
      case 'NOT_FOUND':
        results.notFound.push(result);
        console.log(` ✗ NOT FOUND`);
        break;
      case 'NULL_RESULT':
        results.nullResult.push(result);
        console.log(` ✗ NULL RESULT`);
        break;
      case 'INVALID_RESULT':
        results.invalidResult.push(result);
        console.log(` ✗ INVALID`);
        break;
      case 'RUNTIME_ERROR':
        results.runtimeError.push(result);
        console.log(` ✗ ERROR: ${result.error.substring(0, 60)}`);
        break;
      case 'TEST_ERROR':
        results.testError.push(result);
        console.log(` ✗ TEST ERROR: ${result.error.substring(0, 60)}`);
        break;
    }
  }

  // Print summary
  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total operations: ${demos.length}`);
  console.log(`✓ Working: ${results.ok.length} (${((results.ok.length / demos.length) * 100).toFixed(1)}%)`);
  console.log(`✗ Not found: ${results.notFound.length}`);
  console.log(`✗ Null results: ${results.nullResult.length}`);
  console.log(`✗ Invalid results: ${results.invalidResult.length}`);
  console.log(`✗ Runtime errors: ${results.runtimeError.length}`);
  console.log(`✗ Test errors: ${results.testError.length}`);

  // Print details of failures
  if (results.runtimeError.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log(`RUNTIME ERRORS (${results.runtimeError.length})`);
    console.log('='.repeat(80));
    const byCategory = {};
    for (const r of results.runtimeError) {
      if (!byCategory[r.category]) byCategory[r.category] = [];
      byCategory[r.category].push(r);
    }
    for (const [cat, items] of Object.entries(byCategory)) {
      console.log(`\n${cat}:`);
      for (const item of items) {
        console.log(`  - ${item.name} (${item.id})`);
        console.log(`    ${item.error}`);
      }
    }
  }

  if (results.notFound.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log(`NOT FOUND (${results.notFound.length})`);
    console.log('='.repeat(80));
    for (const r of results.notFound) {
      console.log(`  - ${r.name} (${r.id}): ${r.error}`);
    }
  }

  // Save results
  fs.writeFileSync('operation-test-results.json', JSON.stringify(results, null, 2));
  console.log('\n✓ Detailed results saved to operation-test-results.json');
}

main().catch(console.error);
