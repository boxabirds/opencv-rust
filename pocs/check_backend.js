import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan']
  });
  const page = await browser.newPage();
  
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('GPU') || text.includes('CPU') || text.includes('backend')) {
      console.log('PAGE:', text);
    }
  });
  
  await page.goto('http://localhost:8000/blur_test.html');
  await new Promise(resolve => setTimeout(resolve, 4000));
  
  const backend = await page.evaluate(async () => {
    const module = await import('/pkg/opencv_rust.js');
    return {
      backend: module.getBackend?.() || 'unknown',
      resolved: module.getResolvedBackend?.() || 'unknown',
      gpuAvailable: module.isGpuAvailable?.() || false
    };
  });
  
  console.log('\nBackend info:', JSON.stringify(backend, null, 2));
  
  await browser.close();
})();
