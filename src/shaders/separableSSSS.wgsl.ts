/**
 * WGSL Separable Screen-Space Subsurface Scattering (SSSS) Compute Shader.
 * Performs two-pass separable convolution (Horizontal & Vertical) with depth-aware bilateral weighting.
 */

export const separableSSSSShader = /* wgsl */ `
struct SSSSUniforms {
  dir: vec2<f32>,              // (1.0, 0.0) for horizontal pass, (0.0, 1.0) for vertical pass
  viewportSize: vec2<f32>,     // screen dimensions in pixels (width, height)
  sssWidth: f32,               // world-space scattering radius
  maxDepthError: f32,          // bilateral depth rejection threshold
  cameraNear: f32,
  cameraFar: f32,
  cameraFov: f32,
  sampleCount: u32,
};

@group(0) @binding(0) var<uniform> uniforms: SSSSUniforms;
@group(0) @binding(1) var inColorTex: texture_2d<f32>;
@group(0) @binding(2) var inDepthTex: texture_2d<f32>;
@group(0) @binding(3) var colorSampler: sampler;
@group(0) @binding(4) var<storage, read> kernelOffsets: array<f32>;
@group(0) @binding(5) var<storage, read> kernelWeights: array<vec4<f32>>;
@group(0) @binding(6) var outColorTex: texture_storage_2d<rgba16float, write>;

// Convert raw non-linear depth to linear view-space depth
fn linearizeDepth(depth: f32, near: f32, far: f32) -> f32 {
  return (near * far) / (far - depth * (far - near));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let coords = vec2<i32>(id.xy);
  let dims = vec2<i32>(uniforms.viewportSize);

  if (coords.x >= dims.x || coords.y >= dims.y) {
    return;
  }

  let texCoord = (vec2<f32>(coords) + 0.5) / uniforms.viewportSize;

  // Center sample
  let centerColor = textureLoad(inColorTex, coords, 0);
  let rawCenterDepth = textureLoad(inDepthTex, coords, 0).r;
  
  // Background sky rejection
  if (rawCenterDepth >= 0.9999) {
    textureStore(outColorTex, coords, centerColor);
    return;
  }

  let centerLinearDepth = linearizeDepth(rawCenterDepth, uniforms.cameraNear, uniforms.cameraFar);

  // Calculate screen-space pixel projection of sssWidth
  let distanceToPlane = 0.5 * uniforms.viewportSize.y / tan(0.5 * uniforms.cameraFov);
  let projectedRadius = (uniforms.sssWidth / max(centerLinearDepth, 0.001)) * distanceToPlane;

  // Step vector in UV space
  let stepUv = uniforms.dir * (projectedRadius / uniforms.viewportSize);

  var accumulatedColor = vec3<f32>(0.0);
  var totalWeight = vec3<f32>(0.0);

  let nSamples = uniforms.sampleCount;
  for (var i = 0u; i < nSamples; i = i + 1u) {
    let offsetVal = kernelOffsets[i];
    let weight = kernelWeights[i].rgb;

    let sampleUv = texCoord + offsetVal * stepUv;
    let sampleCoords = vec2<i32>(clamp(sampleUv * uniforms.viewportSize, vec2<f32>(0.0), uniforms.viewportSize - 1.0));

    let sampleColor = textureLoad(inColorTex, sampleCoords, 0).rgb;
    let sampleRawDepth = textureLoad(inDepthTex, sampleCoords, 0).r;
    let sampleLinearDepth = linearizeDepth(sampleRawDepth, uniforms.cameraNear, uniforms.cameraFar);

    // Bilateral depth rejection: discard samples across geometric silhouettes
    let depthDelta = abs(centerLinearDepth - sampleLinearDepth);
    let depthWeight = 1.0 / (1.0 + (depthDelta / max(uniforms.maxDepthError, 0.001)) * 50.0);

    let effectiveWeight = weight * depthWeight;
    accumulatedColor = accumulatedColor + sampleColor * effectiveWeight;
    totalWeight = totalWeight + effectiveWeight;
  }

  // Renormalize to prevent darkening near borders
  let finalRgb = accumulatedColor / max(totalWeight, vec3<f32>(1e-5));
  textureStore(outColorTex, coords, vec4<f32>(finalRgb, centerColor.a));
}
`;
