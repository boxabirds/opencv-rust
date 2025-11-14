/**
 * Test script to systematically check which operations work
 */

import init, { WasmMat, initGpu } from '../../pkg/opencv_rust.js';
import { demos } from './src/demos/demoRegistry.js';
import fs from 'fs';

const SAMPLE_IMAGE = './samples/640x480.png';

async function testOperation(demo) {
  try {
    // Import the operation dynamically
    const module = await import('../../pkg/opencv_rust.js');
    const operation = module[demo.id];

    if (!operation) {
      return {
        id: demo.id,
        name: demo.name,
        status: 'NOT_EXPORTED',
        error: 'Function not found in WASM module'
      };
    }

    // Create a simple test image
    const testMat = WasmMat.from_rgba_data(
      new Uint8Array(640 * 480 * 4).fill(128),
      640,
      480
    );

    // Get default params
    const params = {};
    if (demo.params) {
      demo.params.forEach(param => {
        params[param.id] = param.default;
      });
    }

    // Try to run the operation
    let result;
    try {
      // Different operations have different signatures
      if (demo.params && demo.params.length > 0) {
        // Operation with parameters
        const args = demo.params.map(p => params[p.id]);
        result = operation(testMat, ...args);
      } else {
        // Operation without parameters
        result = operation(testMat);
      }

      // Check if result is valid
      if (result && (result.width || result.cols)) {
        testMat.free();
        if (result.free) result.free();
        return {
          id: demo.id,
          name: demo.name,
          status: 'OK',
          error: null
        };
      } else {
        testMat.free();
        return {
          id: demo.id,
          name: demo.name,
          status: 'INVALID_RESULT',
          error: 'Operation returned invalid result'
        };
      }
    } catch (e) {
      testMat.free();
      return {
        id: demo.id,
        name: demo.name,
        status: 'RUNTIME_ERROR',
        error: e.message
      };
    }
  } catch (e) {
    return {
      id: demo.id,
      name: demo.name,
      status: 'IMPORT_ERROR',
      error: e.message
    };
  }
}

async function main() {
  console.log('Initializing WASM module...');
  await init();

  console.log('Initializing GPU...');
  try {
    await initGpu();
    console.log('GPU initialized successfully');
  } catch (e) {
    console.log('GPU initialization failed (ok for CPU-only operations):', e.message);
  }

  console.log(`\nTesting ${demos.length} operations...\n`);

  const results = {
    ok: [],
    notExported: [],
    runtimeError: [],
    invalidResult: [],
    importError: []
  };

  for (const demo of demos) {
    const result = await testOperation(demo);

    switch (result.status) {
      case 'OK':
        results.ok.push(result);
        console.log(`✓ ${result.name}`);
        break;
      case 'NOT_EXPORTED':
        results.notExported.push(result);
        console.log(`✗ ${result.name} - NOT EXPORTED`);
        break;
      case 'RUNTIME_ERROR':
        results.runtimeError.push(result);
        console.log(`✗ ${result.name} - RUNTIME ERROR: ${result.error}`);
        break;
      case 'INVALID_RESULT':
        results.invalidResult.push(result);
        console.log(`✗ ${result.name} - INVALID RESULT`);
        break;
      case 'IMPORT_ERROR':
        results.importError.push(result);
        console.log(`✗ ${result.name} - IMPORT ERROR: ${result.error}`);
        break;
    }
  }

  // Print summary
  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total operations: ${demos.length}`);
  console.log(`✓ Working: ${results.ok.length}`);
  console.log(`✗ Not exported: ${results.notExported.length}`);
  console.log(`✗ Runtime errors: ${results.runtimeError.length}`);
  console.log(`✗ Invalid results: ${results.invalidResult.length}`);
  console.log(`✗ Import errors: ${results.importError.length}`);

  // Print detailed list of missing operations
  if (results.notExported.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('MISSING OPERATIONS (NOT EXPORTED)');
    console.log('='.repeat(80));
    results.notExported.forEach(r => {
      console.log(`- ${r.id} (${r.name})`);
    });
  }

  if (results.runtimeError.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RUNTIME ERRORS');
    console.log('='.repeat(80));
    results.runtimeError.forEach(r => {
      console.log(`- ${r.id} (${r.name}): ${r.error}`);
    });
  }

  // Save results to file
  fs.writeFileSync('test-results.json', JSON.stringify(results, null, 2));
  console.log('\n✓ Results saved to test-results.json');
}

main().catch(console.error);
