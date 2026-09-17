import { CPUReferenceDiffusion, ImageBuffer } from '../src/core/CPUReferenceDiffusion.js';
import { PRESET_SKIN } from '../src/types/index.js';

interface BenchResult {
  resolution: string;
  totalPixels: number;
  sampleCount: number;
  elapsedMs: number;
  megapixelsPerSec: number;
}

function runBenchmark(width: number, height: number, sampleCount: number = 17, iterations: number = 3): BenchResult {
  const totalPixels = width * height;
  const colorData = new Float32Array(totalPixels * 4);
  const depthData = new Float32Array(totalPixels);

  // Fill dummy test image
  for (let i = 0; i < totalPixels; i++) {
    colorData[i * 4 + 0] = Math.random();
    colorData[i * 4 + 1] = Math.random();
    colorData[i * 4 + 2] = Math.random();
    colorData[i * 4 + 3] = 1.0;
    depthData[i] = 1.0 + (i % width) * 0.05;
  }

  const imgBuf: ImageBuffer = { width, height, data: colorData };
  const config = {
    width,
    height,
    sampleCount,
    sssWidth: 1.2,
    maxDepthError: 0.05,
    cameraNear: 0.1,
    cameraFar: 100.0,
    cameraFov: 45 * Math.PI / 180,
  };

  // Warmup
  CPUReferenceDiffusion.filter(imgBuf, depthData, PRESET_SKIN, config);

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    CPUReferenceDiffusion.filter(imgBuf, depthData, PRESET_SKIN, config);
  }
  const avgElapsed = (performance.now() - start) / iterations;

  return {
    resolution: `${width}x${height}`,
    totalPixels,
    sampleCount,
    elapsedMs: Number(avgElapsed.toFixed(2)),
    megapixelsPerSec: Number(((totalPixels / (avgElapsed / 1000)) / 1e6).toFixed(2)),
  };
}

console.log('⚡ SEPARABLE SSSS BILATERAL DIFFUSION BENCHMARK');
console.log('=============================================================================');
console.log('| Resolution | Total Pixels | Samples | Elapsed (ms) | Throughput (MPix/sec) |');
console.log('-----------------------------------------------------------------------------');

const configs = [
  { w: 128, h: 128, s: 17 },
  { w: 256, h: 256, s: 17 },
  { w: 512, h: 512, s: 17 },
];

for (const c of configs) {
  const res = runBenchmark(c.w, c.h, c.s, 2);
  console.log(
    `| ${res.resolution.padEnd(10)} | ${res.totalPixels.toLocaleString().padStart(12)} | ${res.sampleCount.toString().padStart(7)} | ${res.elapsedMs.toFixed(2).padStart(12)} | ${res.megapixelsPerSec.toFixed(2).padStart(21)} |`
  );
}

console.log('=============================================================================');
console.log('✔ Benchmark completed successfully.');
