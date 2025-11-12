import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for OpenCV.js parity tests
 *
 * Enables headless WebGPU testing using SwiftShader software renderer
 * for consistent, reproducible results across CI and local environments.
 */
export default defineConfig({
  testDir: './',
  testMatch: '**/*.spec.js',

  // Test timeout
  timeout: 60 * 1000, // 60 seconds per test

  // Reporting
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results.json' }],
    ['list']
  ],

  // Shared settings for all projects
  use: {
    // Base URL for test harness
    baseURL: 'http://localhost:8080',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Trace on first retry
    trace: 'on-first-retry',

    // Video on failure
    video: 'retain-on-failure',
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],

        // Enable WebGPU in headless mode with SwiftShader
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',              // Enable WebGPU API
            '--enable-features=Vulkan',            // GPU backend
            '--use-angle=swiftshader',             // Software GPU (consistent results)
            '--disable-vulkan-fallback-to-gl-for-testing',
            '--enable-webgpu-developer-features',  // Better error messages
            '--disable-web-security',              // Allow local file access
            '--allow-file-access-from-files',      // Required for local WASM
          ],
        },
      },
    },
  ],

  // Web server for test harness
  webServer: {
    command: 'python3 -m http.server 8080',
    port: 8080,
    timeout: 120 * 1000,
    reuseExistingServer: !process.env.CI,
    cwd: '../../',  // Serve from project root to access pkg/ and tests/
  },
});
