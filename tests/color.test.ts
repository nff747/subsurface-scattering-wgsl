import { describe, it, expect } from 'vitest';
import { ColorUtils } from '../src/utils/color';

describe('ColorUtils Photometric Conversions', () => {
  it('should round-trip between sRGB and Linear space', () => {
    const srgb = 0.5;
    const linear = ColorUtils.srgbToLinear(srgb);
    const back = ColorUtils.linearToSrgb(linear);
    expect(back).toBeCloseTo(srgb, 4);
  });

  it('should compute Rec.709 relative luminance accurately', () => {
    const lumWhite = ColorUtils.calcLuminance(1.0, 1.0, 1.0);
    expect(lumWhite).toBeCloseTo(1.0, 4);

    const lumGreen = ColorUtils.calcLuminance(0.0, 1.0, 0.0);
    expect(lumGreen).toBeCloseTo(0.7152, 4);
  });
});
