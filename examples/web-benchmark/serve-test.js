import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 8888;

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  console.log('Request:', req.url);

  let filePath = req.url === '/' ? '/test-nlm-simple.html' : req.url;

  // Handle requests to /pkg/* by serving from ../../pkg
  if (filePath.startsWith('/pkg/')) {
    filePath = path.join(__dirname, '../../', filePath);
  } else {
    filePath = path.join(__dirname, filePath);
  }

  const extname = path.extname(filePath);
  const contentType = mimeTypes[extname] || 'text/plain';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.error('Error reading file:', filePath, err.message);
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    res.writeHead(200, {
      'Content-Type': contentType,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Test server running at http://localhost:${PORT}/`);
});
