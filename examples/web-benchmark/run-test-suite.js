#!/usr/bin/env node
/**
 * Run the automated WASM endpoint test suite using Puppeteer
 */

import puppeteer from 'puppeteer';
import fs from 'fs';

const TEST_URL = 'http://localhost:3002/test-suite.html';
const OUTPUT_FILE = 'wasm-test-results.json';
const TIMEOUT_MS = 900000; // 15 minutes

async function runTestSuite() {
  console.log('================================================================================');
  console.log('AUTOMATED WASM ENDPOINT TEST SUITE');
  console.log('================================================================================');
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

    const page = await browser.newPage();

    // Enable console logging from the page
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('[')) {
        console.log(text);
      }
    });

    // Handle page errors
    page.on('pageerror', error => {
      console.error('❌ Page error:', error.message);
    });

    // Handle crashes
    page.on('error', error => {
      console.error('❌ Page crashed:', error.message);
    });

    console.log(`📄 Loading ${TEST_URL}...`);
    await page.goto(TEST_URL, { waitUntil: 'networkidle0', timeout: 30000 });

    console.log('⏳ Waiting for initialization...');
    await page.waitForFunction(() => {
      const btn = document.getElementById('startBtn');
      return btn && !btn.disabled;
    }, { timeout: 30000 });

    console.log('✓ Test suite initialized');
    console.log('');
    console.log('================================================================================');
    console.log('RUNNING TESTS');
    console.log('================================================================================');
    console.log('');

    // Click the start button
    await page.click('#startBtn');

    // Wait for tests to complete
    try {
      await page.waitForFunction(() => {
        const statusText = document.getElementById('statusText');
        return statusText && statusText.textContent === 'Complete';
      }, { timeout: TIMEOUT_MS });
    } catch (error) {
      // On timeout, get the current status for debugging
      const currentStatus = await page.evaluate(() => {
        const statusEl = document.getElementById('statusText');
        const progressEl = document.getElementById('progress');
        return {
          status: statusEl ? statusEl.textContent : 'unknown',
          progress: progressEl ? progressEl.textContent : 'unknown'
        };
      });
      console.error(`⚠️ Test timed out. Status: ${currentStatus.status}, Progress: ${currentStatus.progress}`);
      throw error;
    }

    console.log('');
    console.log('================================================================================');
    console.log('TESTS COMPLETE');
    console.log('================================================================================');
    console.log('');

    // Extract results
    const results = await page.evaluate(() => {
      return window.testResults;
    });

    if (!results) {
      throw new Error('Failed to retrieve test results from window.testResults');
    }

    // Save results to file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
    console.log(`✓ Results saved to ${OUTPUT_FILE}`);
    console.log('');

    // Print summary
    console.log('================================================================================');
    console.log('SUMMARY');
    console.log('================================================================================');
    console.log(`Total operations:     ${results.summary.total}`);
    console.log(`✓ Passing:            ${results.summary.passing} (${results.summary.passRate}%)`);
    console.log(`✗ Failing:            ${results.summary.failing} (${(100 - results.summary.passRate).toFixed(1)}%)`);
    console.log('');
    console.log('Breakdown:');
    console.log(`  ✓ OK:               ${results.results.ok.length}`);
    console.log(`  ✗ Not exported:     ${results.results.notExported.length}`);
    console.log(`  ✗ Null output:      ${results.results.nullOutput.length}`);
    console.log(`  ✗ Invalid output:   ${results.results.invalidOutput.length}`);
    console.log(`  ✗ Channel mismatch: ${results.results.channelMismatch.length}`);
    console.log(`  ✗ Unimplemented:    ${results.results.unimplemented.length}`);
    console.log(`  ✗ WASM panic:       ${results.results.wasmPanic.length}`);
    console.log(`  ✗ Invalid param:    ${results.results.invalidParam.length}`);
    console.log(`  ✗ Unknown error:    ${results.results.unknownError.length}`);

    // Print failure details
    const failureCategories = [
      { name: 'CHANNEL MISMATCH', items: results.results.channelMismatch },
      { name: 'NOT EXPORTED', items: results.results.notExported },
      { name: 'UNIMPLEMENTED', items: results.results.unimplemented },
      { name: 'WASM PANIC', items: results.results.wasmPanic },
      { name: 'INVALID PARAMETER', items: results.results.invalidParam },
      { name: 'NULL OUTPUT', items: results.results.nullOutput },
      { name: 'INVALID OUTPUT', items: results.results.invalidOutput },
      { name: 'UNKNOWN ERROR', items: results.results.unknownError }
    ];

    for (const category of failureCategories) {
      if (category.items.length > 0) {
        console.log('');
        console.log('================================================================================');
        console.log(`${category.name} (${category.items.length})`);
        console.log('================================================================================');

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
              const errorPreview = item.error.substring(0, 100);
              console.log(`    Error: ${errorPreview}${item.error.length > 100 ? '...' : ''}`);
            }
          }
        }
      }
    }

    // Performance stats
    if (results.results.ok.length > 0) {
      console.log('');
      console.log('================================================================================');
      console.log('PERFORMANCE (passing tests only)');
      console.log('================================================================================');

      const durations = results.results.ok.map(r => r.duration);
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

    console.log('');
    console.log('================================================================================');

    await browser.close();

    // Exit with error code if any tests failed
    process.exit(results.summary.failing > 0 ? 1 : 0);

  } catch (error) {
    console.error('');
    console.error('================================================================================');
    console.error('FATAL ERROR');
    console.error('================================================================================');
    console.error(error.message);
    console.error(error.stack);

    if (browser) {
      await browser.close();
    }

    process.exit(1);
  }
}

runTestSuite();
