import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();

  page.on('console', msg => console.log('PAGE:', msg.text()));

  await page.goto('http://localhost:8000/border_test.html');
  await new Promise(resolve => setTimeout(resolve, 2000));

  const output = await page.locator('#output').innerHTML();
  console.log('\n' + output.replace(/<[^>]*>/g, '\n').replace(/&gt;/g, '>').replace(/&lt;/g, '<'));

  await browser.close();
})();
