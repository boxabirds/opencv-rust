#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

console.log('\n=== Loading test page ===');
await page.goto('http://localhost:3000/test-gaussian.html', { waitUntil: 'networkidle0' });

// Wait for test to complete
await new Promise(resolve => setTimeout(resolve, 5000));

// Check result
const statusText = await page.evaluate(() => {
  const el = document.getElementById('status');
  return el ? el.textContent : 'Status element not found';
});

console.log(`\n=== Final status: ${statusText} ===`);

if (statusText.includes('SUCCESS')) {
  console.log('✓ Test PASSED');
} else {
  console.log('✗ Test FAILED');
}

await browser.close();
