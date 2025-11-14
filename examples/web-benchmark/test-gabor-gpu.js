import puppeteer from 'puppeteer';

const TEST_URL = 'http://localhost:3000/test-suite.html';

async function testGaborGPU() {
  console.log('Starting Gabor Filter GPU test...');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--no-sandbox', '--disable-gpu-sandbox']
  });

  const page = await browser.newPage();

  // Capture console messages
  page.on('console', msg => {
    const text = msg.text();
    console.log(text);
  });

  page.on('pageerror', error => {
    console.error('Page error:', error.message);
  });

  try {
    console.log('Loading page...');
    await page.goto(TEST_URL, { waitUntil: 'networkidle0', timeout: 30000 });

    console.log('Waiting for page to be ready...');
    await page.waitForFunction(() => typeof window.runSingleTest === 'function', { timeout: 30000 });

    console.log('Running Gabor Filter test...');
    const result = await page.evaluate(async () => {
      try {
        // Wait for WASM to load
        if (!window.wasmModule) {
          await new Promise(resolve => {
            const check = setInterval(() => {
              if (window.wasmModule) {
                clearInterval(check);
                resolve();
              }
            }, 100);
          });
        }

        // Run gabor filter test
        const testResult = await window.runSingleTest('gabor_filter');
        return testResult;
      } catch (e) {
        return { error: e.message, stack: e.stack };
      }
    });

    console.log('\n=== GABOR FILTER TEST RESULT ===');
    console.log(JSON.stringify(result, null, 2));

    if (result.error) {
      console.error('Test failed with error:', result.error);
      process.exit(1);
    } else if (result.passed) {
      console.log('✅ TEST PASSED');
      console.log(`GPU Time: ${result.gpuTime?.toFixed(2) || 'N/A'}ms`);
      console.log(`CPU Time: ${result.cpuTime?.toFixed(2) || 'N/A'}ms`);
      if (result.speedup) {
        console.log(`Speedup: ${result.speedup.toFixed(2)}x`);
      }
      process.exit(0);
    } else {
      console.error('❌ TEST FAILED');
      process.exit(1);
    }

  } catch (error) {
    console.error('Error during test:', error.message);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

testGaborGPU().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
