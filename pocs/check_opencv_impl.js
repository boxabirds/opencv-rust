import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();

  await page.goto('http://localhost:8000/extract_opencv_impl.html');
  await new Promise(resolve => setTimeout(resolve, 2000));

  const output = await page.locator('#output').innerText();
  console.log(output);

  await browser.close();
})();
