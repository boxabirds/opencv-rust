import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto('http://localhost:8000/trace_border.html');
  await new Promise(resolve => setTimeout(resolve, 500));

  const output = await page.locator('#output').innerText();
  console.log(output);

  await browser.close();
})();
