/**
 * Headless CPU Reference Subsurface Diffusion Implementation.
 * Provides verifiable separable bilateral convolution for testing and CPU fallbacks.
 */

import { SSSSProfile, SSSSConfig } from '../types';
import { KernelGenerator } from '../math/kernel';

export interface ImageBuffer {
  width: number;
  height: number;
  data: Float32Array; // RGBA float values [0.0, 1.0]
}

export class CPUReferenceDiffusion {
  /**
   * Convolves an RGBA image buffer with separable SSSS kernel and bilateral depth rejection.
   */
  public static filter(
    colorBuffer: ImageBuffer,
    depthBuffer: Float32Array, // Linear depth per pixel
    profile: SSSSProfile,
    config: SSSSConfig
  ): ImageBuffer {
    const { width, height } = colorBuffer;
    const kernel = KernelGenerator.generate1DKernel(profile, config.sampleCount);

    const temp = new Float32Array(width * height * 4);
    const output = new Float32Array(width * height * 4);

    // Pass 1: Horizontal Blur (X)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerIdx = y * width + x;
        const centerDepth = depthBuffer[centerIdx];

        if (centerDepth <= 0 || centerDepth >= config.cameraFar * 0.99) {
          temp[centerIdx * 4 + 0] = colorBuffer.data[centerIdx * 4 + 0];
          temp[centerIdx * 4 + 1] = colorBuffer.data[centerIdx * 4 + 1];
          temp[centerIdx * 4 + 2] = colorBuffer.data[centerIdx * 4 + 2];
          temp[centerIdx * 4 + 3] = colorBuffer.data[centerIdx * 4 + 3];
          continue;
        }

        const distanceToPlane = 0.5 * height / Math.tan(0.5 * config.cameraFov);
        const projectedRadius = (config.sssWidth / Math.max(centerDepth, 0.001)) * distanceToPlane;

        let accumR = 0, accumG = 0, accumB = 0;
        let weightR = 0, weightG = 0, weightB = 0;

        for (let s = 0; s < kernel.sampleCount; s++) {
          const offset = kernel.offsets[s];
          const sampleX = Math.round(x + offset * (projectedRadius / width) * width);

          if (sampleX >= 0 && sampleX < width) {
            const sIdx = y * width + sampleX;
            const sDepth = depthBuffer[sIdx];
            const depthDelta = Math.abs(centerDepth - sDepth);
            const depthWeight = 1.0 / (1.0 + (depthDelta / Math.max(config.maxDepthError, 0.001)) * 50.0);

            const wR = kernel.weights[s * 4 + 0] * depthWeight;
            const wG = kernel.weights[s * 4 + 1] * depthWeight;
            const wB = kernel.weights[s * 4 + 2] * depthWeight;

            accumR += colorBuffer.data[sIdx * 4 + 0] * wR;
            accumG += colorBuffer.data[sIdx * 4 + 1] * wG;
            accumB += colorBuffer.data[sIdx * 4 + 2] * wB;

            weightR += wR;
            weightG += wG;
            weightB += wB;
          }
        }

        temp[centerIdx * 4 + 0] = accumR / Math.max(weightR, 1e-5);
        temp[centerIdx * 4 + 1] = accumG / Math.max(weightG, 1e-5);
        temp[centerIdx * 4 + 2] = accumB / Math.max(weightB, 1e-5);
        temp[centerIdx * 4 + 3] = colorBuffer.data[centerIdx * 4 + 3];
      }
    }

    // Pass 2: Vertical Blur (Y)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerIdx = y * width + x;
        const centerDepth = depthBuffer[centerIdx];

        if (centerDepth <= 0 || centerDepth >= config.cameraFar * 0.99) {
          output[centerIdx * 4 + 0] = temp[centerIdx * 4 + 0];
          output[centerIdx * 4 + 1] = temp[centerIdx * 4 + 1];
          output[centerIdx * 4 + 2] = temp[centerIdx * 4 + 2];
          output[centerIdx * 4 + 3] = temp[centerIdx * 4 + 3];
          continue;
        }

        const distanceToPlane = 0.5 * height / Math.tan(0.5 * config.cameraFov);
        const projectedRadius = (config.sssWidth / Math.max(centerDepth, 0.001)) * distanceToPlane;

        let accumR = 0, accumG = 0, accumB = 0;
        let weightR = 0, weightG = 0, weightB = 0;

        for (let s = 0; s < kernel.sampleCount; s++) {
          const offset = kernel.offsets[s];
          const sampleY = Math.round(y + offset * (projectedRadius / height) * height);

          if (sampleY >= 0 && sampleY < height) {
            const sIdx = sampleY * width + x;
            const sDepth = depthBuffer[sIdx];
            const depthDelta = Math.abs(centerDepth - sDepth);
            const depthWeight = 1.0 / (1.0 + (depthDelta / Math.max(config.maxDepthError, 0.001)) * 50.0);

            const wR = kernel.weights[s * 4 + 0] * depthWeight;
            const wG = kernel.weights[s * 4 + 1] * depthWeight;
            const wB = kernel.weights[s * 4 + 2] * depthWeight;

            accumR += temp[sIdx * 4 + 0] * wR;
            accumG += temp[sIdx * 4 + 1] * wG;
            accumB += temp[sIdx * 4 + 2] * wB;

            weightR += wR;
            weightG += wG;
            weightB += wB;
          }
        }

        output[centerIdx * 4 + 0] = accumR / Math.max(weightR, 1e-5);
        output[centerIdx * 4 + 1] = accumG / Math.max(weightG, 1e-5);
        output[centerIdx * 4 + 2] = accumB / Math.max(weightB, 1e-5);
        output[centerIdx * 4 + 3] = temp[centerIdx * 4 + 3];
      }
    }

    return {
      width,
      height,
      data: output,
    };
  }
}
