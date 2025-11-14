#!/usr/bin/env node
import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: false, dumpio: true, slow: 100 });
const page = await browser.newPage();

page.on('console', msg => console.log('[PAGE]', msg.text()));
page.on('pageerror', err => console.error('[ERROR]', err.message));

await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

console.log('\n=== Testing Gaussian Blur via UI click ===');

// Wait for app to load
await page.waitForSelector('button', { timeout: 5000 });
console.log('Page loaded');

// Find and click Gaussian Blur button/link
await page.evaluate(() => {
  const elements = Array.from(document.querySelectorAll('*'));
  const gaussianElement = elements.find(el =>
    el.textContent && el.textContent.includes('Gaussian Blur')
  );
  if (gaussianElement) {
    console.log('Found Gaussian Blur element, clicking...');
    gaussianElement.click();
  } else {
    console.log('Gaussian Blur element not found');
  }
});

// Wait for processing
await new Promise(resolve => setTimeout(resolve, 3000));

console.log('\n✓ UI test complete - check browser window for results');
console.log('Press Ctrl+C to close');

// Keep browser open
await new Promise(resolve => setTimeout(resolve, 30000));
await browser.close();
