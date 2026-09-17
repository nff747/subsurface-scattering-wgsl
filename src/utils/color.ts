/**
 * Photometric & Color Space Utilities for Physical BSSRDF Evaluation.
 * Converts between sRGB, Linear Rec.709, and CIE 1931 XYZ for spectral diffusion mapping.
 */

export class ColorUtils {
  public static srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  public static linearToSrgb(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
  }

  public static rgbToLinear([r, g, b]: [number, number, number]): [number, number, number] {
    return [this.srgbToLinear(r), this.srgbToLinear(g), this.srgbToLinear(b)];
  }

  public static linearToRgb([r, g, b]: [number, number, number]): [number, number, number] {
    return [this.linearToSrgb(r), this.linearToSrgb(g), this.linearToSrgb(b)];
  }

  public static calcLuminance(r: number, g: number, b: number): number {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
}
