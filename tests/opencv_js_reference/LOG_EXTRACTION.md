# Log Extraction Guide

The test harness now has comprehensive logging to help debug initialization issues.

## How Logging Works

All logs are stored in `window.debugLog` array with the following structure:

```javascript
{
  time: 123.45,              // milliseconds since page load
  category: "OpenCV.js",     // log category
  message: "Loading...",     // log message
  level: "log",              // "log", "warn", or "error"
  timestamp: "2025-01-01..." // ISO timestamp
}
```

## Extracting Logs

### Method 1: Automated (In Tests)

Run the debug test which automatically extracts and displays logs:

```bash
cd tests/opencv_js_reference
npm test test_debug.spec.js
```

Look for the "Debug Log (window.debugLog)" section in the output.

### Method 2: Browser Console

1. Open test-harness.html in your browser:
   ```bash
   # Start web server from project root
   python3 -m http.server 8080

   # Open in browser
   open http://localhost:8080/tests/opencv_js_reference/test-harness.html
   ```

2. Open browser DevTools (F12)

3. In the console, run:
   ```javascript
   window.extractDebugLog()
   ```

   This will print all logs in a formatted way.

4. To get the raw log array:
   ```javascript
   window.debugLog
   ```

5. To save logs to a file:
   ```javascript
   copy(JSON.stringify(window.debugLog, null, 2))
   ```
   Then paste into a text file.

### Method 3: Programmatic Extraction

In Playwright tests, you can extract logs like this:

```javascript
const debugLog = await page.evaluate(() => window.debugLog);
console.log(JSON.stringify(debugLog, null, 2));
```

## What's Logged

### Initialization Phase
- Browser information (user agent, platform, WebGPU availability)
- Memory usage at each stage
- Timing information for each step

### OpenCV.js Loading
- Script tag status
- Polling attempts (every 100ms)
- Progress updates (every 2 seconds)
- Callback status
- Final state (success or timeout)

### WASM Loading
- Module import timing
- WASM initialization timing
- GPU initialization attempts
- Module export names
- Memory usage before/after each step

### Errors
- All errors with stack traces
- Global errors
- Unhandled promise rejections

## Log Categories

- `Startup` - Initial startup
- `Init` - Main initialization
- `OpenCV.js` - OpenCV.js loading
- `WASM` - WASM module loading
- `Test Harness` - Status updates
- `Memory` - Memory usage tracking
- `Details` - Additional details
- `Global Error` - Uncaught errors
- `Unhandled Rejection` - Promise rejections
- `Fatal` - Fatal errors

## Filtering Logs

To filter logs by category:

```javascript
// In browser console
window.debugLog.filter(log => log.category === 'OpenCV.js')

// In Node.js/test
debugLog.filter(log => log.category === 'OpenCV.js')
```

To filter by level:

```javascript
// Only errors
window.debugLog.filter(log => log.level === 'error')

// Errors and warnings
window.debugLog.filter(log => ['error', 'warn'].includes(log.level))
```

## Example Output

```
📝 [0.70ms] [Startup] Calling initialize()...
📝 [1.20ms] [Init] === INITIALIZATION START ===
📝 [1.50ms] [Init] Browser: Mozilla/5.0...
📝 [2.00ms] [Init] WebGPU available: true
📝 [2.50ms] [Memory] Initialization start: 9.54MB used / 9.54MB total
📝 [3.00ms] [Test Harness] Loading OpenCV.js...
📝 [3.20ms] [Init] === PHASE 1: Loading OpenCV.js ===
📝 [3.50ms] [OpenCV.js] === Starting OpenCV.js load process ===
📝 [3.80ms] [OpenCV.js] Not loaded yet, checking script tag...
📝 [4.00ms] [OpenCV.js] Script tag found: http://localhost:8080/cache/opencv.js
📝 [4.20ms] [OpenCV.js] window.cv type: undefined
📝 [4.50ms] [OpenCV.js] Starting polling for cv object (check every 100ms for 30s)...
📝 [2004.50ms] [OpenCV.js] Still polling... 2000ms elapsed
📝 [2004.70ms] [OpenCV.js]   cv exists: false
❌ [30004.50ms] [OpenCV.js] ✗ TIMEOUT after 30 seconds
❌ [30004.80ms] [OpenCV.js] Final state - cv exists: false
```

## Troubleshooting

### No logs showing up

Check if `window.debugLog` exists:
```javascript
typeof window.debugLog  // should be "object"
window.debugLog.length  // should be > 0
```

### Logs are empty

The page might not have loaded properly. Check:
- Browser console for errors
- Network tab for failed requests
- Make sure test-harness.html is the correct version

### extractDebugLog() not defined

The page hasn't finished loading the script. Wait a moment and try again, or check the console for errors.

## Saving Logs for Sharing

To share logs with others:

1. Extract logs:
   ```javascript
   window.extractDebugLog()
   ```

2. Copy all console output (Cmd+A, Cmd+C in console)

3. Or save as JSON:
   ```javascript
   copy(JSON.stringify({
     userAgent: navigator.userAgent,
     debugLog: window.debugLog,
     initSteps: window.initSteps,
     finalState: {
       testHarnessReady: window.testHarnessReady,
       opencv_js_ready: window.opencv_js_ready,
       opencv_rust_ready: window.opencv_rust_ready,
       cvExists: typeof cv !== 'undefined',
       cvMatExists: typeof cv !== 'undefined' && typeof cv.Mat !== 'undefined'
     }
   }, null, 2))
   ```

4. Paste into a file named `debug-logs.json`

## Next Steps

If logs show timeout issues:
1. Check if OpenCV.js file is accessible at `/cache/opencv.js`
2. Verify WASM file is built at `/pkg/opencv_rust.js`
3. Check for JavaScript errors in the console
4. Try in different browser or headless vs headed mode
5. Share logs with the development team
