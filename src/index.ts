/**
 * Subsurface Scattering WGSL
 * Real-Time Separable Screen-Space Subsurface Scattering (SSSS) & Translucency Engine in WebGPU / WGSL
 * @packageDocumentation
 */

export * from './types';
export * from './math/kernel';
export * from './core/SSSSPipeline';
export * from './core/CPUReferenceDiffusion';
export * from './core/ThreeSSSSPass';

export { separableSSSSShader } from './shaders/separableSSSS.wgsl';
export { translucencyShader } from './shaders/translucency.wgsl';
