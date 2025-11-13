import { chromium } from '@playwright/test';

async function testKsize(ksize, sigma) {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();
  await page.goto('http://localhost:8000/blur_test.html');
  
  await page.fill('#ksize', ksize.toString());
  await page.fill('#sigma', sigma.toString());
  await page.click('button');
  
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  const status = await page.locator('#status').innerText();
  const lines = status.split('\n');
  const diffLine = lines.find(l => l.includes('Different pixels'));
  
  await browser.close();
  return diffLine;
}

(async () => {
  const tests = [
    [3, 0.5],
    [5, 1.5],
    [7, 2.0],
    [9, 3.0]
  ];
  
  console.log('Testing different kernel sizes:\n');
  for (const [ksize, sigma] of tests) {
    const result = await testKsize(ksize, sigma);
    console.log(`ksize=${ksize}, σ=${sigma}: ${result}`);
  }
})();
