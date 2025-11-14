import puppeteer from 'puppeteer';

async function testNlm() {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  // Capture console and errors
  page.on('console', msg => console.log('PAGE:', msg.text()));
  page.on('pageerror', err => console.log('ERROR:', err.message));

  await page.goto('http://localhost:3000/test-suite.html', { waitUntil: 'networkidle0' });

  await page.waitForFunction(() => window.WasmMat, { timeout: 10000 });

  const result = await page.evaluate(async () => {
    try {
      // Import module
      const module = await import('../../pkg/opencv_rust.js');
      await module.default();

      // Create test image
      const width = 100, height = 100, channels = 1;
      const data = new Uint8Array(width * height * channels);
      for (let i = 0; i < data.length; i++) {
        data[i] = Math.floor(Math.random() * 256);
      }

      const mat = new module.WasmMat(data, width, height, channels);

      console.log('Testing nlmDenoising...');
      const result = await module.nlmDenoising(mat, 10, 7, 21);

      mat.free();
      result.free();

      return { success: true };
    } catch (e) {
      return { success: false, error: e.message, stack: e.stack };
    }
  });

  console.log('Result:', result);
  await browser.close();
}

testNlm().catch(console.error);
