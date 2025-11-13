#!/usr/bin/env python3
"""
HTTP server with CORS headers for SharedArrayBuffer support.
Required for wasm-bindgen-rayon threading to work.
"""
import http.server
import socketserver
from pathlib import Path

PORT = 8080

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required headers for SharedArrayBuffer (needed for wasm-threading)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        # Also allow CORS for resources
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()

if __name__ == '__main__':
    # Change to project root (two directories up)
    import os
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    print(f"Serving from: {project_root}")
    print(f"Server running at http://localhost:{PORT}/")
    print("Required CORS headers enabled for wasm-threading")

    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
