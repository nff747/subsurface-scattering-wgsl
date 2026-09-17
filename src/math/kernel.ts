/**
 * Separable Screen-Space Subsurface Scattering (SSSS) Kernel Generator.
 * Based on Jimenez et al. (SIGGRAPH 2009 / GPU Pro) and Burley (SIGGRAPH 2015).
 * Generates an energy-conserving 1D Gaussian sum kernel approximated across RGB channels.
 */

import { SSSSProfile, DiffusionKernel } from '../types';

export class KernelGenerator {
  /**
   * Generates a 1D discrete sampling kernel from an SSSS profile.
   * Uses importance sampling of Gaussian standard deviations to allocate sample offsets.
   *
   * @param profile Subsurface scattering color falloff and distance profile.
   * @param sampleCount Number of samples (must be odd, e.g. 11, 17, 25).
   * @returns DiffusionKernel containing 1D offsets and normalized vec4 RGB weights.
   */
  public static generate1DKernel(profile: SSSSProfile, sampleCount: number = 17): DiffusionKernel {
    if (sampleCount % 2 === 0) {
      sampleCount += 1;
    }

    const offsets = new Float32Array(sampleCount);
    // Weights: RGBA packed (RGB = weight, A = 0 padding)
    const weights = new Float32Array(sampleCount * 4);

    const halfSamples = Math.floor(sampleCount / 2);
    const range = 3.0; // Standard deviations range

    // 6-term Gaussian sum coefficients for human skin / realistic dipole diffusion
    // [variance, RGB weight fractions]
    const gaussians = [
      { variance: 0.0064, weight: [0.233, 0.455, 0.649] },
      { variance: 0.0484, weight: [0.100, 0.336, 0.344] },
      { variance: 0.1870, weight: [0.118, 0.198, 0.0] },
      { variance: 0.5670, weight: [0.113, 0.007, 0.007] },
      { variance: 1.9900, weight: [0.358, 0.004, 0.0] },
      { variance: 7.4100, weight: [0.078, 0.0, 0.0] }
    ];

    let sumR = 0;
    let sumG = 0;
    let sumB = 0;

    // Center sample
    offsets[halfSamples] = 0.0;

    // Sample positions along 1D line: non-linear distribution denser near origin
    for (let i = 0; i < sampleCount; i++) {
      const idx = i - halfSamples;
      const sign = idx < 0 ? -1 : 1;
      // Cubic spacing concentration near zero
      const t = Math.abs(idx) / halfSamples;
      const offset = sign * Math.pow(t, 2.0) * range;
      offsets[i] = offset;

      // Evaluate Gaussian sum at offset r for each channel
      let rWeight = 0;
      let gWeight = 0;
      let bWeight = 0;

      const r2 = offset * offset;
      for (const g of gaussians) {
        // Apply profile scattering distance scaling
        const varR = g.variance * profile.scatteringDistance[0];
        const varG = g.variance * profile.scatteringDistance[1];
        const varB = g.variance * profile.scatteringDistance[2];

        const gR = (1.0 / Math.sqrt(2.0 * Math.PI * varR)) * Math.exp(-r2 / (2.0 * varR));
        const gG = (1.0 / Math.sqrt(2.0 * Math.PI * varG)) * Math.exp(-r2 / (2.0 * varG));
        const gB = (1.0 / Math.sqrt(2.0 * Math.PI * varB)) * Math.exp(-r2 / (2.0 * varB));

        rWeight += g.weight[0] * gR;
        gWeight += g.weight[1] * gG;
        bWeight += g.weight[2] * gB;
      }

      weights[i * 4 + 0] = rWeight;
      weights[i * 4 + 1] = gWeight;
      weights[i * 4 + 2] = bWeight;
      weights[i * 4 + 3] = 0.0;

      sumR += rWeight;
      sumG += gWeight;
      sumB += bWeight;
    }

    // Strictly normalize kernel so sum of weights per channel = 1.0 (Energy Conservation)
    for (let i = 0; i < sampleCount; i++) {
      weights[i * 4 + 0] /= sumR;
      weights[i * 4 + 1] /= sumG;
      weights[i * 4 + 2] /= sumB;
    }

    return {
      sampleCount,
      offsets,
      weights,
    };
  }

  /**
   * Calculates world-space scattering blur radius projected into screen-space pixels.
   *
   * @param sssWidth World space scattering radius in millimeters.
   * @param linearDepth Linear eye-space distance from camera in millimeters.
   * @param cameraFov Vertical field of view in radians.
   * @param viewportHeight Screen resolution height in pixels.
   */
  public static calculateProjectedRadius(
    sssWidth: number,
    linearDepth: number,
    cameraFov: number,
    viewportHeight: number
  ): number {
    if (linearDepth <= 1e-4) return 0;
    const distanceToPlane = 0.5 * viewportHeight / Math.tan(0.5 * cameraFov);
    return (sssWidth / linearDepth) * distanceToPlane;
  }
}
