#!/usr/bin/env node
/**
 * Check if WASM needs rebuilding by comparing source file timestamps
 * with the built WASM output.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '../../..');
const pkgDir = path.join(rootDir, 'pkg');
const wasmFile = path.join(pkgDir, 'opencv_rust_bg.wasm');

/**
 * Get the most recent modification time from a directory recursively
 */
function getMostRecentMtime(dir, extensions = ['.rs', '.toml']) {
  let latestMtime = 0;

  function walk(currentDir) {
    const files = fs.readdirSync(currentDir);

    for (const file of files) {
      const fullPath = path.join(currentDir, file);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        // Skip target and node_modules
        if (file === 'target' || file === 'node_modules' || file === '.git') {
          continue;
        }
        walk(fullPath);
      } else if (stat.isFile()) {
        // Check if file has relevant extension
        const ext = path.extname(file);
        if (extensions.includes(ext)) {
          if (stat.mtimeMs > latestMtime) {
            latestMtime = stat.mtimeMs;
          }
        }
      }
    }
  }

  walk(dir);
  return latestMtime;
}

/**
 * Check if git working directory is dirty for Rust files
 */
function isGitDirty() {
  try {
    const output = execSync('git status --porcelain -- src/ Cargo.toml Cargo.lock', {
      cwd: rootDir,
      encoding: 'utf8'
    });
    return output.trim().length > 0;
  } catch (e) {
    // If git command fails, assume dirty to be safe
    return true;
  }
}

/**
 * Main check function
 */
function checkWasmFreshness() {
  // Check if WASM output exists
  if (!fs.existsSync(wasmFile)) {
    console.log('🔨 WASM build missing, rebuild needed');
    return false;
  }

  const wasmStat = fs.statSync(wasmFile);
  const wasmMtime = wasmStat.mtimeMs;

  // Check if git working directory is dirty
  if (isGitDirty()) {
    console.log('🔨 Git working directory has Rust changes, rebuild needed');
    return false;
  }

  // Check source file timestamps
  const srcDir = path.join(rootDir, 'src');
  const srcMtime = getMostRecentMtime(srcDir, ['.rs']);

  if (srcMtime > wasmMtime) {
    console.log('🔨 Source files newer than WASM, rebuild needed');
    return false;
  }

  // Check Cargo.toml/Cargo.lock
  const cargoToml = path.join(rootDir, 'Cargo.toml');
  const cargoLock = path.join(rootDir, 'Cargo.lock');

  const cargoTomlMtime = fs.existsSync(cargoToml)
    ? fs.statSync(cargoToml).mtimeMs
    : 0;
  const cargoLockMtime = fs.existsSync(cargoLock)
    ? fs.statSync(cargoLock).mtimeMs
    : 0;

  if (cargoTomlMtime > wasmMtime) {
    console.log('🔨 Cargo.toml newer than WASM, rebuild needed');
    return false;
  }

  if (cargoLockMtime > wasmMtime) {
    console.log('🔨 Cargo.lock newer than WASM, rebuild needed');
    return false;
  }

  // All checks passed - WASM is fresh
  console.log('✅ WASM build is up to date, skipping rebuild');
  return true;
}

// Run check and exit with appropriate code
const isFresh = checkWasmFreshness();
process.exit(isFresh ? 0 : 1);
