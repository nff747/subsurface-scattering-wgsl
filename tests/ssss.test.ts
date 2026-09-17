import { describe, it, expect } from 'vitest';
import {
  KernelGenerator,
  SSSSPipeline,
  CPUReferenceDiffusion,
  ThreeSSSSPass,
  PRESET_SKIN,
  PRESET_MARBLE,
  PRESET_JADE,
  PRESET_WAX,
  PRESET_MILK,
  separableSSSSShader,
  translucencyShader,
} from '../src/index';

describe('Jimenez SSSS Gaussian Kernel Generator', () => {
  it('should generate energy-conserving normalized weights summing to 1.0 for RGB', () => {
    const sampleCount = 17;
    const kernel = KernelGenerator.generate1DKernel(PRESET_SKIN, sampleCount);

    expect(kernel.sampleCount).toBe(sampleCount);
    expect(kernel.offsets.length).toBe(sampleCount);
    expect(kernel.weights.length).toBe(sampleCount * 4);

    let sumR = 0, sumG = 0, sumB = 0;
    for (let i = 0; i < sampleCount; i++) {
      sumR += kernel.weights[i * 4 + 0];
      sumG += kernel.weights[i * 4 + 1];
      sumB += kernel.weights[i * 4 + 2];
    }

    expect(sumR).toBeCloseTo(1.0, 4);
    expect(sumG).toBeCloseTo(1.0, 4);
    expect(sumB).toBeCloseTo(1.0, 4);
  });

  it('should have symmetric sample offsets centered at zero', () => {
    const kernel = KernelGenerator.generate1DKernel(PRESET_MARBLE, 15);
    const mid = Math.floor(kernel.sampleCount / 2);

    expect(kernel.offsets[mid]).toBeCloseTo(0.0, 5);

    for (let i = 0; i < mid; i++) {
      const left = kernel.offsets[i];
      const right = kernel.offsets[kernel.sampleCount - 1 - i];
      expect(left).toBeCloseTo(-right, 5);
    }
  });

  it('should accurately project world scattering radius to screen pixels', () => {
    // 1.2mm scattering width at 1000mm distance, 45deg FOV, 1080p
    const pixels = KernelGenerator.calculateProjectedRadius(
      1.2,
      1000.0,
      45 * Math.PI / 180,
      1080
    );

    expect(pixels).toBeGreaterThan(0.5);
    expect(pixels).toBeLessThan(10.0);
  });
});

describe('Material Presets & Profile Configurations', () => {
  it('should define distinct scattering profiles for skin, marble, jade, and wax', () => {
    expect(PRESET_SKIN.scatteringDistance[0]).toBeGreaterThan(PRESET_SKIN.scatteringDistance[2]); // Red scatters further in skin
    expect(PRESET_JADE.scatteringDistance[1]).toBeGreaterThan(PRESET_JADE.scatteringDistance[0]); // Green scatters further in jade
    expect(PRESET_MARBLE.scatteringDistance[0]).toBeCloseTo(PRESET_MARBLE.scatteringDistance[1], 0.2); // Marble is isotropic white
    expect(PRESET_MILK.scatteringDistance[0]).toBeGreaterThan(3.0); // High scattering distance
  });
});

describe('SSSSPipeline WebGPU Coordinator', () => {
  it('should correctly format uniform buffer layout with direction and camera params', () => {
    const pipeline = new SSSSPipeline(null, {
      width: 1920,
      height: 1080,
      sssWidth: 1.5,
    });

    const uniformsH = pipeline.buildUniformData([1.0, 0.0]);
    expect(uniformsH[0]).toBe(1.0);
    expect(uniformsH[1]).toBe(0.0);
    expect(uniformsH[2]).toBe(1920);
    expect(uniformsH[3]).toBe(1080);
    expect(uniformsH[4]).toBe(1.5);

    const uniformsV = pipeline.buildUniformData([0.0, 1.0]);
    expect(uniformsV[0]).toBe(0.0);
    expect(uniformsV[1]).toBe(1.0);
  });
});

describe('CPUReferenceDiffusion Separable Bilateral Filtering', () => {
  it('should preserve sharp depth edge discontinuity while diffusing planar region', () => {
    const width = 16;
    const height = 16;
    const colorData = new Float32Array(width * height * 4);
    const depthData = new Float32Array(width * height);

    // Left half: Foreground plane at depth = 1.0, White color
    // Right half: Background wall at depth = 50.0, Black color
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (x < 8) {
          colorData[idx * 4 + 0] = 1.0;
          colorData[idx * 4 + 1] = 1.0;
          colorData[idx * 4 + 2] = 1.0;
          colorData[idx * 4 + 3] = 1.0;
          depthData[idx] = 1.0;
        } else {
          colorData[idx * 4 + 0] = 0.0;
          colorData[idx * 4 + 1] = 0.0;
          colorData[idx * 4 + 2] = 0.0;
          colorData[idx * 4 + 3] = 1.0;
          depthData[idx] = 50.0;
        }
      }
    }

    const filtered = CPUReferenceDiffusion.filter(
      { width, height, data: colorData },
      depthData,
      PRESET_SKIN,
      {
        width,
        height,
        sampleCount: 11,
        sssWidth: 0.5,
        maxDepthError: 0.05,
        cameraNear: 0.1,
        cameraFar: 100.0,
        cameraFov: 45 * Math.PI / 180,
      }
    );

    // Due to bilateral depth rejection, background should NOT bleed into foreground
    const fgSample = filtered.data[(8 * width + 6) * 4 + 0];
    expect(fgSample).toBeGreaterThan(0.7);

    // Background should remain close to 0.0
    const bgSample = filtered.data[(8 * width + 10) * 4 + 0];
    expect(bgSample).toBeLessThan(0.2);
  });
});

describe('WGSL Shader Source Verification', () => {
  it('should contain expected compute workgroups and symbols', () => {
    expect(separableSSSSShader).toContain('@compute');
    expect(separableSSSSShader).toContain('@workgroup_size(16, 16)');
    expect(separableSSSSShader).toContain('linearizeDepth');
    expect(separableSSSSShader).toContain('maxDepthError');

    expect(translucencyShader).toContain('@compute');
    expect(translucencyShader).toContain('thicknessScale');
    expect(translucencyShader).toContain('translucencyColor');
  });
});
