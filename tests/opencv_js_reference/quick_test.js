import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newContext().then(c => c.newPage());

  await page.goto('http://localhost:8787/save_bytes.html');

  // Wait for completion
  await page.waitForFunction(() => {
    return document.getElementById('status').textContent.includes('PNG images saved');
  }, { timeout: 30000 });

  // Get the stats
  const stats = await page.evaluate(() => document.getElementById('status').innerHTML);
  console.log('\n' + stats.replace(/<[^>]*>/g, '\n').replace(/\n+/g, '\n'));

  await browser.close();
})();
