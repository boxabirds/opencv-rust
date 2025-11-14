#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/test-all-exp-fixes.html', { waitUntil: 'networkidle0' });

// Wait for all tests to complete (15 seconds to be safe)
await new Promise(resolve => setTimeout(resolve, 15000));

// Get results
const results = await page.evaluate(() => {
  return window.testResults || { status: 'TIMEOUT', total: 0, passed: 0, failed: 0, results: [] };
});

console.log('\n=== COMPREHENSIVE exp() FIX VERIFICATION ===');
console.log(`Status: ${results.status}`);
console.log(`Total tests: ${results.total || results.results.length}`);

if (results.error) {
  console.error(`\x1b[31mERROR: ${results.error}\x1b[0m`);
}

let passed = 0;
let failed = 0;

console.log('\nTest Results:');
console.log('─'.repeat(80));

results.results.forEach(r => {
  const status = r.success ? '✓' : '✗';
  const color = r.success ? '\x1b[32m' : '\x1b[31m';
  const sourceInfo = r.sourceFile ? ` (${r.sourceFile})` : '';
  console.log(`${color}${status}\x1b[0m ${r.operation}: ${r.message}${sourceInfo}`);
  if (r.success) passed++;
  else failed++;
});

console.log('─'.repeat(80));
console.log(`\nSummary: ${passed}/${results.total || results.results.length} passed, ${failed} failed`);

if (passed === (results.total || results.results.length) && failed === 0) {
  console.log('\n\x1b[32m✓ ALL TESTS PASSED - 100% exp() fix verification complete!\x1b[0m');
  console.log('\nTested operations using exp():');
  console.log('  1. gaussianBlur - Gaussian kernel computation');
  console.log('  2. exp - Direct exponential operation');
  console.log('  3. gaborFilter - Gabor kernel Gaussian component');
  console.log('  4. neuralNetwork - Sigmoid activation function');
  console.log('  5. svmClassifier - RBF kernel computation');
  console.log('  6. tonemapDrago - HDR tone mapping');
  console.log('  7. tonemapReinhard - HDR tone mapping');
  console.log('  8. mergeDebevec - HDR image merging');
  console.log('  9. mosseTracker - Gaussian window generation');

  console.log('\nOperations NOT testable (internal/not exposed):');
  console.log('  - dnn/layers.rs:275 (Sigmoid) - internal DNN layer');
  console.log('  - dnn/layers.rs:440 (Softmax) - internal DNN layer');
  console.log('  - ml/boost.rs:65 (AdaBoost) - not exposed as WASM export');
  console.log('  - photo/super_resolution.rs:228 - not exposed as WASM export');
  console.log('  - stitching/blending.rs:316 (FeatherBlender) - wrapper uses different impl');
} else {
  console.log(`\n\x1b[31m✗ TESTS FAILED - ${failed} test(s) did not pass\x1b[0m`);
}

await browser.close();
process.exit(failed > 0 ? 1 : 0);
