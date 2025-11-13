import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();

  await page.goto('http://localhost:8000/blur_test.html');

  // Test multiple kernel sizes
  const tests = [[3, 0.5], [5, 1.5], [7, 2.0], [11, 3.0]];

  console.log('Kernel Size | Sigma | Error %');
  console.log('------------|-------|--------');

  for (const [ksize, sigma] of tests) {
    await page.fill('#ksize', ksize.toString());
    await page.fill('#sigma', sigma.toString());
    await page.click('button');
    await new Promise(resolve => setTimeout(resolve, 2000));

    const status = await page.locator('#status').innerText();
    const match = status.match(/Different pixels: [\d,]+ \(([\d.]+)%\)/);
    const errorPct = match ? match[1] : 'N/A';

    console.log(`${ksize.toString().padStart(11)} | ${sigma.toString().padStart(5)} | ${errorPct.padStart(6)}`);
  }

  await browser.close();
})();
