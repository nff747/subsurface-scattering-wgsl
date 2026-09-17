/**
 * WebGPU SSSS Pipeline Coordinator.
 * Manages GPU compute pipelines, uniform/storage buffers, and two-pass separable dispatches.
 */

import { SSSSConfig, SSSSProfile, DiffusionKernel, PRESET_SKIN } from '../types';
import { KernelGenerator } from '../math/kernel';
import { separableSSSSShader } from '../shaders/separableSSSS.wgsl';

export class SSSSPipeline {
  public device: GPUDevice | null;
  public config: SSSSConfig;
  public profile: SSSSProfile;
  public kernel: DiffusionKernel;

  constructor(device: GPUDevice | null = null, config?: Partial<SSSSConfig>, profile: SSSSProfile = PRESET_SKIN) {
    this.device = device;
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
    this.profile = profile;
    this.kernel = KernelGenerator.generate1DKernel(this.profile, this.config.sampleCount);
  }

  /**
   * Updates current SSSS material profile (e.g. Skin, Marble, Jade) and recomputes kernel.
   */
  public setProfile(profile: SSSSProfile): void {
    this.profile = profile;
    this.kernel = KernelGenerator.generate1DKernel(this.profile, this.config.sampleCount);
  }

  /**
   * Generates uniform array buffer for WGSL shader binding.
   *
   * @param dir Direction vector (1, 0) for horizontal or (0, 1) for vertical.
   */
  public buildUniformData(dir: [number, number]): Float32Array {
    const buffer = new Float32Array(12);
    buffer[0] = dir[0];
    buffer[1] = dir[1];
    buffer[2] = this.config.width;
    buffer[3] = this.config.height;
    buffer[4] = this.config.sssWidth;
    buffer[5] = this.config.maxDepthError;
    buffer[6] = this.config.cameraNear;
    buffer[7] = this.config.cameraFar;
    buffer[8] = this.config.cameraFov;
    // sampleCount as uint32 bitcasted
    const u32View = new Uint32Array(buffer.buffer);
    u32View[9] = this.config.sampleCount;
    // buffer[10], buffer[11] are padding
    return buffer;
  }

  /**
   * Dispatches two-pass separable SSSS blur using provided WebGPU command encoder.
   */
  public recordPasses(
    encoder: GPUCommandEncoder,
    computePipeline: GPUComputePipeline,
    horizontalBindGroup: GPUBindGroup,
    verticalBindGroup: GPUBindGroup
  ): void {
    const workgroupsX = Math.ceil(this.config.width / 16);
    const workgroupsY = Math.ceil(this.config.height / 16);

    // Pass 1: Horizontal blur
    const pass1 = encoder.beginComputePass({ label: 'SSSS_Horizontal_Pass' });
    pass1.setPipeline(computePipeline);
    pass1.setBindGroup(0, horizontalBindGroup);
    pass1.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass1.end();

    // Pass 2: Vertical blur
    const pass2 = encoder.beginComputePass({ label: 'SSSS_Vertical_Pass' });
    pass2.setPipeline(computePipeline);
    pass2.setBindGroup(0, verticalBindGroup);
    pass2.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass2.end();
  }
}
