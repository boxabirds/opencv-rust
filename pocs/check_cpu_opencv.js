import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();

  page.on('console', msg => console.log('PAGE:', msg.text()));

  await page.goto('http://localhost:8000/test_cpu_vs_opencv.html');
  await new Promise(resolve => setTimeout(resolve, 3000));

  const status = await page.locator('#status').innerHTML();
  console.log('\n' + status.replace(/<br>/g, '\n').replace(/<[^>]*>/g, ''));

  await browser.close();
})();
