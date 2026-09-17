/**
 * Separable Screen-Space Subsurface Scattering (SSSS) & Translucency Types.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface SSSSProfile {
  name: string;
  falloff: [number, number, number];           // Per-channel RGB falloff factor (shape of diffusion profile)
  scatteringDistance: [number, number, number];// Mean free path per channel in millimeters
  diffuseReflectance: [number, number, number];// Diffuse surface color multiplier
  translucencyColor: [number, number, number]; // Back-scattering tint
}

export interface DiffusionKernel {
  sampleCount: number;
  offsets: Float32Array;                       // Normalized 1D sample offsets along blur axis
  weights: Float32Array;                       // RGB weights per sample (packed as vec4 with alpha padding)
}

export interface SSSSConfig {
  width: number;
  height: number;
  sampleCount: number;                         // e.g. 11, 17, or 25 samples
  sssWidth: number;                            // Global scattering scale in world units (mm)
  maxDepthError: number;                       // Bilateral depth rejection threshold to prevent halo bleeding
  cameraNear: number;
  cameraFar: number;
  cameraFov: number;                           // Vertical FOV in radians
}

export interface TranslucencyParams {
  distortion: number;                          // Normal distortion factor (0.0 to 1.0)
  power: number;                               // Exponent of transmission highlight
  scale: number;                               // Transmission intensity multiplier
  ambient: number;                             // Minimum ambient transmission floor
  thicknessScale: number;                      // Scaling for object depth difference
}

export const PRESET_SKIN: SSSSProfile = {
  name: 'Human Skin (Caucasian)',
  falloff: [1.0, 0.37, 0.3],
  scatteringDistance: [1.44, 0.55, 0.25],
  diffuseReflectance: [0.85, 0.72, 0.65],
  translucencyColor: [0.98, 0.22, 0.15],
};

export const PRESET_MARBLE: SSSSProfile = {
  name: 'White Marble',
  falloff: [0.85, 0.85, 0.85],
  scatteringDistance: [2.19, 2.05, 1.62],
  diffuseReflectance: [0.93, 0.91, 0.88],
  translucencyColor: [0.88, 0.82, 0.75],
};

export const PRESET_JADE: SSSSProfile = {
  name: 'Imperial Jade',
  falloff: [0.15, 0.9, 0.2],
  scatteringDistance: [0.65, 2.50, 0.75],
  diffuseReflectance: [0.25, 0.75, 0.35],
  translucencyColor: [0.35, 0.95, 0.45],
};

export const PRESET_WAX: SSSSProfile = {
  name: 'Candle Wax',
  falloff: [0.9, 0.7, 0.3],
  scatteringDistance: [2.80, 2.10, 1.20],
  diffuseReflectance: [0.92, 0.88, 0.76],
  translucencyColor: [0.95, 0.75, 0.45],
};

export const PRESET_MILK: SSSSProfile = {
  name: 'Whole Milk',
  falloff: [0.95, 0.95, 0.9],
  scatteringDistance: [3.45, 3.20, 2.80],
  diffuseReflectance: [0.95, 0.94, 0.91],
  translucencyColor: [0.98, 0.97, 0.95],
};
