/**
 * WGSL Back-Surface Translucency & Light Transmission Shader.
 * Computes forward scattering through thin geometry (ears, nose, leaves, fingers, marble fins).
 */

export const translucencyShader = /* wgsl */ `
struct TranslucencyUniforms {
  lightDir: vec3<f32>,
  distortion: f32,             // Normal distortion
  lightColor: vec3<f32>,
  power: f32,                  // Transmission sharpness exponent
  translucencyColor: vec3<f32>,
  scale: f32,                  // Intensity multiplier
  ambient: f32,
  thicknessScale: f32,
};

@group(0) @binding(0) var<uniform> params: TranslucencyUniforms;
@group(0) @binding(1) var inNormalTex: texture_2d<f32>;
@group(0) @binding(2) var inThicknessTex: texture_2d<f32>;
@group(0) @binding(3) var outTranslucencyTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let coords = vec2<i32>(id.xy);
  let dims = textureDimensions(inNormalTex);

  if (coords.x >= dims.x || coords.y >= dims.y) {
    return;
  }

  let normalSample = textureLoad(inNormalTex, coords, 0).xyz;
  let normal = normalize(normalSample * 2.0 - 1.0);
  let thickness = textureLoad(inThicknessTex, coords, 0).r * params.thicknessScale;

  // View direction towards camera (orthogonal / standard eye Z)
  let viewDir = vec3<f32>(0.0, 0.0, 1.0);

  // Distort light direction using surface normal
  let lTrans = normalize(params.lightDir + normal * params.distortion);
  let vDotL = max(0.0, dot(viewDir, -lTrans));

  // Exponential attenuation through volume thickness
  let attenuation = exp(-thickness * 4.0);
  let intensity = (pow(vDotL, params.power) * params.scale + params.ambient) * attenuation;

  let transRgb = params.lightColor * params.translucencyColor * intensity;
  textureStore(outTranslucencyTex, coords, vec4<f32>(transRgb, 1.0));
}
`;
