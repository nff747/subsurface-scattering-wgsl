/**
 * Three.js SSSS Post-Processing Render Pass Adapter.
 * Integrates Separable SSSS convolution with Three.js WebGLRenderer / WebGPURenderer pipelines.
 */

import { SSSSProfile, SSSSConfig, PRESET_SKIN } from '../types';
import { KernelGenerator } from '../math/kernel';

export interface ThreeRenderTargetLike {
  texture: any;
  depthTexture?: any;
  setSize: (w: number, h: number) => void;
  dispose: () => void;
}

export class ThreeSSSSPass {
  public profile: SSSSProfile;
  public config: SSSSConfig;
  public enabled: boolean = true;
  public kernelOffsets: Float32Array;
  public kernelWeights: Float32Array;

  constructor(config?: Partial<SSSSConfig>, profile: SSSSProfile = PRESET_SKIN) {
    this.profile = profile;
    this.config = {
      width: config?.width ?? 1920,
      height: config?.height ?? 1080,
      sampleCount: config?.sampleCount ?? 17,
      sssWidth: config?.sssWidth ?? 1.2,
      maxDepthError: config?.maxDepthError ?? 0.05,
      cameraNear: config?.cameraNear ?? 0.1,
      cameraFar: config?.cameraFar ?? 100.0,
      cameraFov: config?.cameraFov ?? (45 * Math.PI / 180),
    };

    const kernel = KernelGenerator.generate1DKernel(this.profile, this.config.sampleCount);
    this.kernelOffsets = kernel.offsets;
    this.kernelWeights = kernel.weights;
  }

  /**
   * Sets the active subsurface scattering material profile.
   */
  public setProfile(profile: SSSSProfile): void {
    this.profile = profile;
    const kernel = KernelGenerator.generate1DKernel(this.profile, this.config.sampleCount);
    this.kernelOffsets = kernel.offsets;
    this.kernelWeights = kernel.weights;
  }

  /**
   * Updates viewport resolution.
   */
  public setSize(width: number, height: number): void {
    this.config.width = width;
    this.config.height = height;
  }

  /**
   * Generates custom shader uniform definition object for Three.js ShaderMaterial.
   */
  public getUniforms(): Record<string, { value: any }> {
    return {
      sssWidth: { value: this.config.sssWidth },
      maxDepthError: { value: this.config.maxDepthError },
      cameraNear: { value: this.config.cameraNear },
      cameraFar: { value: this.config.cameraFar },
      cameraFov: { value: this.config.cameraFov },
      sampleCount: { value: this.config.sampleCount },
      kernelOffsets: { value: this.kernelOffsets },
      kernelWeights: { value: this.kernelWeights },
    };
  }
}
