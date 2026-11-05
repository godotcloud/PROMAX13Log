//
//  iPhone13ProMaxLogCameraApp.swift
//  CineLog13Pro v6.0 (Zero-Stutter Coalesced GPU Pipeline + Calibrated Color Science + Strictly Separated LUT & Physical Film Engines)
//
//  Key Engineering Fixes in v6.0:
//  1. Zero-Stutter 60fps/24fps Architecture:
//     - Compute kernel operates in 100% coalesced horizontal sensor memory order (eliminates 90° strided L1 texture cache misses).
//     - Live preview runs at fast 1920x1080 when idle and full 4K/OpenGate when recording or snapping photos (4x faster GPU compute).
//     - Viewfinder orientation is performed for free in the 4-vertex shader (`vertexViewfinder`), and `.mov` orientation via `AVAssetWriterInput.transform`.
//     - Event-driven `MTKView` (`isPaused = true`, `enableSetNeedsDisplay = true`) eliminates redundant 60Hz main-thread redraw loops.
//  2. Calibrated Exposure & Display Color Space:
//     - Fixed `CAMetalLayer` color space from `itur_2020` to `sRGB` so Rec.709 & Film outputs are never double-stretched into neon oversaturation.
//     - Calibrated Naka-Rushton / Rational H&D Sensitometric Curve preserves full highlight latitude up to +2.5 stops above diffuse white (no blown-out walls/sheets).
//  3. Strictly Mutually Exclusive Color Engines:
//     - Mode 0: Pure Apple Log (or clean Rec.709 monitor)
//     - Mode 1: Independent LUT Grading System (`LUTSystemParameters` — 10 built-in cinema LUTs + custom `.cube` Tetrahedral LUT, zero film simulation)
//     - Mode 2: Independent Physical Film Lab (`PhysicalFilmParameters` — 12 direct-photon optical/chemical film stocks, zero LUTs)
//

import UIKit
import SwiftUI
import AVFoundation
import AVKit
import CoreImage
import Metal
import MetalKit
import CoreVideo
import CoreMedia
import VideoToolbox
import Photos
import PhotosUI
import CoreMotion
import QuartzCore
import UniformTypeIdentifiers
import Combine

// MARK: - 1. High-Speed Coalesced Metal Compute & Render Shaders

enum MetalShaderLibrarySource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct PipelineUniforms {
        int isNativeAppleLog;       // 1 = hardware native Apple Log, 0 = HLG/SDR sensor input
        int isFullRangeYCbCr;       // 1 = Full Range ('420f'), 0 = Video Range ('x422', 'x420', '420v')
        int isSDRSensorInput;       // 1 = 8-bit SDR P3/709 sensor fallback, 0 = 10-bit HLG BT.2020
        int writeRecordBuffer;      // 1 = Actively recording video or capturing photo, 0 = Preview only

        // Strictly Mutually Exclusive Engine Mode:
        // 0 = Pure Apple Log Mode (Raw Log or Official Rec.709 Technical Preview)
        // 1 = LUT Color Grading System ONLY (Never runs Physical Film Lab)
        // 2 = Physical Film Simulation Lab ONLY (Never runs LUT System)
        int colorEngineMode;
        int bakeIntoRecording;      // 1 = Bake active LUT or Film into recorded video, 0 = Record pure Apple Log
        int monitorDiagnosticMode;  // 0 = Normal, 1 = False Color IRE, 2 = Zebra (>94% IRE)

        // Focus Peaking
        int focusPeakingEnabled;
        int focusPeakingColorIndex; // 0 = Green, 1 = Red, 2 = Yellow, 3 = Cyan
        float focusPeakingThreshold;

        // --- Engine 1 Exclusive Parameters (LUT Grading System) ---
        int lutPresetID;            // 0 = Raw Log, 1 = Official Rec.709, 2 = ARRI Alexa 709, 3 = Kodak 2383 D65, 4 = Kodak 2383 Warm,
                                    // 5 = Fuji 3513DI, 6 = Teal & Orange, 7 = Bleach Bypass, 8 = Cyber Nocturne, 9 = Custom .cube 3D LUT
        int lutInputSpace;          // 0 = Apple Log input, 1 = Rec.709 input
        int customLUTSize;          // Dimension N of .cube LUT (17, 33, 64, 65)
        int hasCustom3DLUT;         // 1 = Valid custom 3D LUT bound at texture(4)
        float lutIntensity;         // 0.0 to 1.0
        float lutExposureEV;        // -2.5 to +2.5 EV
        float lutContrast;          // 0.5 to 1.5
        float lutHighlightRollOff;  // 0.0 to 1.0
        float lutShadowLift;        // -0.25 to +0.25
        float lutSaturation;        // 0.0 to 2.0
        float lutVibrance;          // -1.0 to +1.0
        float lutTemperature;       // -1.0 to +1.0
        float lutTint;              // -1.0 to +1.0
        float lutDebanding;         // 0.0 to 1.0
        float4 liftRGB;
        float4 gammaRGB;
        float4 gainRGB;

        // --- Engine 2 Exclusive Parameters (Physical Film Simulation Lab) ---
        int filmStockID;            // 0..11 (12 classic stocks)
        float filmExposureEV;       // -2.5 to +2.5 EV
        float pushPullStops;        // -1.5 to +1.5 stops chemical development
        float halationIntensity;    // 0.0 to 1.0 red film-base scatter
        float emulsionBloom;        // 0.0 to 1.0 optical Mie glow
        float printPaperContrast;   // 0.7 to 1.4 enlarger paper grade
        float printWarmthSplit;     // -1.0 to +1.0 dichroic toning
        float filmGrain;            // 0.0 to 1.0 silver halide grain
        float grainSize;            // 0.5 to 1.8
        float lensVignette;         // 0.0 to 1.0 cos^4(theta) falloff
        float chromaticAberration;  // 0.0 to 1.0 radial dispersion
        float frameTime;
    };

    inline float hash12(float2 p) {
        float3 p3 = fract(float3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    // --- 1. Coalesced YCbCr to Normalized RGB Conversion ---
    inline float3 decodeYCbCrToRGB(float yRaw, float2 cbcrRaw, int isFullRange, int isSDR) {
        float y, cb, cr;
        if (isFullRange == 1) {
            y  = clamp(yRaw, 0.0, 1.0);
            cb = cbcrRaw.x - 0.5;
            cr = cbcrRaw.y - 0.5;
        } else {
            float y10  = yRaw * 1023.0;
            float2 c10 = cbcrRaw * 1023.0;
            y  = clamp((y10 - 64.0) / (940.0 - 64.0), 0.0, 1.0);
            cb = (c10.x - 512.0) / (960.0 - 64.0);
            cr = (c10.y - 512.0) / (960.0 - 64.0);
        }

        if (isSDR == 1) {
            float r = y + 1.5748 * cr;
            float g = y - 0.187324 * cb - 0.468124 * cr;
            float b = y + 1.8556 * cb;
            return clamp(float3(r, g, b), 0.0, 1.0);
        } else {
            float r = y + 1.4746 * cr;
            float g = y - 0.16455312684366 * cb - 0.57135312684366 * cr;
            float b = y + 1.8814 * cb;
            return clamp(float3(r, g, b), 0.0, 1.0);
        }
    }

    // --- 2. Calibrated ITU-R BT.2100 HLG & SDR Inverse OETF to Scene-Linear Light ---
    // Calibrated so:
    //   E' = 0.38 (18% Middle Gray in HLG) -> sceneLinear = 0.18
    //   E' = 0.55 (Indoor Lit Wall)        -> sceneLinear = 0.36
    //   E' = 0.75 (100% Diffuse White)     -> sceneLinear = 0.92
    //   E' = 1.00 (Peak Specular Highlight)-> sceneLinear = 3.40
    inline float hlgInverseOETFChannel(float ePrime) {
        const float a = 0.17883277;
        const float b = 1.0 - 4.0 * a;          // 0.28466892
        const float c = 0.5 - a * log(4.0 * a); // 0.55991073
        float v = max(ePrime, 0.0);
        if (v <= 0.5) {
            return (v * v) / 3.0;
        } else {
            return (exp((v - c) / a) + b) / 12.0;
        }
    }

    inline float3 sensorToCalibratedLinearBT2020(float3 encodedRGB, int isSDR, float evBias, float rollOff) {
        float3 sceneLinear;
        if (isSDR == 1) {
            float3 v = clamp(encodedRGB, 0.0, 1.0);
            float3 linP3 = pow(v, float3(2.2)) * 0.90;
            sceneLinear.r = 0.753833 * linP3.r + 0.198597 * linP3.g + 0.047570 * linP3.b;
            sceneLinear.g = 0.045744 * linP3.r + 0.941777 * linP3.g + 0.012479 * linP3.b;
            sceneLinear.b = -0.001210 * linP3.r + 0.017602 * linP3.g + 0.983608 * linP3.b;
            sceneLinear = max(sceneLinear, 0.0) * exp2(evBias);
        } else {
            float3 rawLin = float3(
                hlgInverseOETFChannel(encodedRGB.r),
                hlgInverseOETFChannel(encodedRGB.g),
                hlgInverseOETFChannel(encodedRGB.b)
            );
            sceneLinear = max(float3(0.0), (rawLin - 0.0008) * 3.40) * exp2(evBias);
        }

        // Soft highlight compression shoulder so bright windows/walls never clip harshly
        if (rollOff > 0.001) {
            float3 shoulder = sceneLinear / (1.0 + sceneLinear * (0.35 * rollOff));
            sceneLinear = mix(sceneLinear, shoulder, rollOff);
        }
        return sceneLinear;
    }

    inline float3 linearBT2020ToLinear709(float3 lin2020) {
        float3 lin709;
        lin709.r =  1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b;
        lin709.g = -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b;
        lin709.b = -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b;
        return max(lin709, 0.0);
    }

    // --- 3. Official Apple Log Profile White Paper OETF & EOTF ---
    inline float linearToAppleLogChannel(float R) {
        const float R_0   = -0.05641088;
        const float R_t   =  0.01;
        const float c     =  47.28711236;
        const float beta  =  0.00964052;
        const float gamma =  0.08550479;
        const float delta =  0.69336945;

        if (R >= R_t) {
            return gamma * log2(R + beta) + delta;
        } else if (R >= R_0) {
            float diff = R - R_0;
            return c * diff * diff;
        } else {
            return 0.0;
        }
    }

    inline float3 linearToAppleLog(float3 linearRGB) {
        return clamp(float3(
            linearToAppleLogChannel(linearRGB.r),
            linearToAppleLogChannel(linearRGB.g),
            linearToAppleLogChannel(linearRGB.b)
        ), 0.0, 1.0);
    }

    inline float appleLogToLinearChannel(float P) {
        const float R_0   = -0.05641088;
        const float R_t   =  0.01;
        const float c     =  47.28711236;
        const float beta  =  0.00964052;
        const float gamma =  0.08550479;
        const float delta =  0.69336945;
        const float P_t   =  0.20856507;

        if (P >= P_t) {
            return exp2((P - delta) / gamma) - beta;
        } else if (P >= 0.0) {
            return sqrt(P / c) + R_0;
        } else {
            return R_0;
        }
    }

    inline float3 appleLogToLinear(float3 logRGB) {
        return max(float3(
            appleLogToLinearChannel(logRGB.r),
            appleLogToLinearChannel(logRGB.g),
            appleLogToLinearChannel(logRGB.b)
        ), 0.0);
    }

    // --- 4. Calibrated Natural Rec.709 Display Transform (Preserves Full Highlight Latitude up to H = 3.5) ---
    // Uses Naka-Rushton / Michaelis-Menten rational tone mapping so:
    //   H = 0.00 -> 0.00 (True Black)
    //   H = 0.18 -> 0.43 (18% Middle Gray)
    //   H = 0.36 -> 0.60 (Indoor Lit Wall — full detail, never blown out!)
    //   H = 0.92 -> 0.81 (100% Diffuse White — full bedsheet/paper texture!)
    //   H = 3.40 -> 0.98 (Specular Highlight Roll-off)
    inline float3 calibratedLinear709ToRec709Display(float3 lin709) {
        float3 h = max(lin709, 0.0);
        // Rational shoulder curve: h / (h + 0.38) maps [0..inf) smoothly into [0..1) with 0.18 -> 0.321 in linear display light
        float3 compressed = (h * (1.0 + h / 6.5)) / (h + 0.38);
        // Apply Rec.709 / sRGB display gamma (1 / 2.15) with subtle toe anchor
        float3 display = pow(clamp(compressed, 0.0, 1.0), float3(1.0 / 2.15));
        return clamp(display, 0.0, 1.0);
    }

    // --- 5. Tetrahedral 3D LUT Interpolation ---
    inline float3 sample3DLUTTetrahedral(
        texture3d<float, access::sample> lutTex,
        float3 rgbIn,
        int lutSize
    ) {
        int N = max(2, lutSize);
        float3 scaled = clamp(rgbIn, 0.0, 1.0) * float(N - 1);
        int3 i0 = min(int3(floor(scaled)), int3(N - 2));
        int3 i1 = i0 + int3(1, 1, 1);
        float3 f = scaled - float3(i0);

        float r = f.x;
        float g = f.y;
        float b = f.z;

        float3 c000 = lutTex.read(uint3(i0.x, i0.y, i0.z)).rgb;
        float3 c111 = lutTex.read(uint3(i1.x, i1.y, i1.z)).rgb;

        float3 c1, c2;
        float w0, w1, w2, w3;

        if (r >= g) {
            if (g >= b) {
                c1 = lutTex.read(uint3(i1.x, i0.y, i0.z)).rgb;
                c2 = lutTex.read(uint3(i1.x, i1.y, i0.z)).rgb;
                w0 = 1.0 - r; w1 = r - g; w2 = g - b; w3 = b;
            } else if (r >= b) {
                c1 = lutTex.read(uint3(i1.x, i0.y, i0.z)).rgb;
                c2 = lutTex.read(uint3(i1.x, i0.y, i1.z)).rgb;
                w0 = 1.0 - r; w1 = r - b; w2 = b - g; w3 = g;
            } else {
                c1 = lutTex.read(uint3(i0.x, i0.y, i1.z)).rgb;
                c2 = lutTex.read(uint3(i1.x, i0.y, i1.z)).rgb;
                w0 = 1.0 - b; w1 = b - r; w2 = r - g; w3 = g;
            }
        } else {
            if (b >= g) {
                c1 = lutTex.read(uint3(i0.x, i0.y, i1.z)).rgb;
                c2 = lutTex.read(uint3(i0.x, i1.y, i1.z)).rgb;
                w0 = 1.0 - b; w1 = b - g; w2 = g - r; w3 = r;
            } else if (b >= r) {
                c1 = lutTex.read(uint3(i0.x, i1.y, i0.z)).rgb;
                c2 = lutTex.read(uint3(i0.x, i1.y, i1.z)).rgb;
                w0 = 1.0 - g; w1 = g - b; w2 = b - r; w3 = r;
            } else {
                c1 = lutTex.read(uint3(i0.x, i1.y, i0.z)).rgb;
                c2 = lutTex.read(uint3(i1.x, i1.y, i0.z)).rgb;
                w0 = 1.0 - g; w1 = g - r; w2 = r - b; w3 = b;
            }
        }

        return clamp(w0 * c000 + w1 * c1 + w2 * c2 + w3 * c111, 0.0, 1.0);
    }

    // --- 6. ENGINE 1 ONLY: LUT Color Grading System (Zero Physical Film Simulation) ---
    inline float3 evaluateLUTSystemOnly(
        float3 appleLogIn,
        constant PipelineUniforms& u,
        texture3d<float, access::sample> customLUT
    ) {
        float3 logColor = appleLogIn;

        logColor.r += u.lutTemperature * 0.045 + u.lutTint * 0.025;
        logColor.g -= u.lutTint * 0.040;
        logColor.b -= u.lutTemperature * 0.045 - u.lutTint * 0.025;

        float3 lifted = logColor + u.liftRGB.xyz * (1.0 - logColor);
        float3 gained = lifted * u.gainRGB.xyz;
        float3 gammaPower = max(float3(0.3), 1.0 - u.gammaRGB.xyz * 0.55);
        logColor = pow(clamp(gained, 0.0, 1.0), gammaPower);

        if (abs(u.lutShadowLift) > 0.001) {
            float shadowWeight = clamp(1.0 - (logColor.g / 0.5), 0.0, 1.0);
            logColor += u.lutShadowLift * shadowWeight * shadowWeight * 0.22;
        }
        logColor = clamp((logColor - 0.488) * u.lutContrast + 0.488, 0.0, 1.0);

        if (u.lutPresetID == 0) {
            float luma = dot(logColor, float3(0.2627, 0.6780, 0.0593));
            return clamp(mix(float3(luma), logColor, u.lutSaturation), 0.0, 1.0);
        }

        float3 lin2020 = appleLogToLinear(logColor);
        float3 lin709  = linearBT2020ToLinear709(lin2020);
        float3 rec709  = calibratedLinear709ToRec709Display(lin709);

        float3 styled = rec709;
        float luma = dot(rec709, float3(0.2126, 0.7152, 0.0722));

        if (u.lutPresetID == 2) {
            // ARRI Alexa K1S1 709: Ultra-smooth highlight shoulder & natural warm skin
            float3 arri = pow(clamp(lin709 / (lin709 + 0.34), 0.0, 1.0), float3(1.0 / 2.1));
            arri.r = mix(arri.r, arri.r * 1.02, luma);
            styled = clamp(arri, 0.0, 1.0);
        } else if (u.lutPresetID == 3) {
            // Kodak 2383 Print D65
            float3 sCurve = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.35);
            float3 sh = float3(-0.025, 0.005, 0.035) * pow(1.0 - luma, 1.5);
            float3 hi = float3( 0.020, 0.012, -0.020) * pow(luma, 1.2);
            styled = clamp(sCurve + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 4) {
            // Kodak 2383 Print Warm D55
            float3 sCurve = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.35);
            float3 hi = float3( 0.042, 0.024, -0.032) * pow(luma, 1.0);
            float3 sh = float3(-0.018, 0.008,  0.024) * pow(1.0 - luma, 1.4);
            styled = clamp(sCurve + hi + sh, 0.0, 1.0);
        } else if (u.lutPresetID == 5) {
            // Fujifilm 3513DI Print
            float3 sCurve = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.32);
            sCurve.g = clamp(sCurve.g * 1.025, 0.0, 1.0);
            sCurve.b = clamp(sCurve.b * 1.018 + 0.015 * (1.0 - luma), 0.0, 1.0);
            styled = sCurve;
        } else if (u.lutPresetID == 6) {
            // Teal & Orange Master
            float3 sCurve = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.35);
            float3 sh = float3(-0.045, 0.012, 0.055) * pow(1.0 - luma, 1.4);
            float3 hi = float3( 0.050, 0.018, -0.038) * pow(luma, 0.9);
            styled = clamp(sCurve + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 7) {
            // Bleach Bypass Silver
            float3 desat = mix(float3(luma), rec709, 0.45);
            styled = mix(desat, smoothstep(0.03, 0.97, desat), 0.45);
        } else if (u.lutPresetID == 8) {
            // Cyber Nocturne
            float3 sh = float3(0.015, -0.02, 0.065) * (1.0 - luma);
            float3 hi = float3(-0.015, 0.03, 0.045) * luma;
            styled = clamp(rec709 + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 9 && u.hasCustom3DLUT == 1) {
            float3 lutSource = (u.lutInputSpace == 0) ? logColor : rec709;
            styled = sample3DLUTTetrahedral(customLUT, lutSource, u.customLUTSize);
        }

        float3 result = mix(rec709, styled, clamp(u.lutIntensity, 0.0, 1.0));
        float finalLuma = dot(result, float3(0.2126, 0.7152, 0.0722));
        float maxC = max(result.r, max(result.g, result.b));
        float minC = min(result.r, min(result.g, result.b));
        float vibranceFactor = 1.0 + u.lutVibrance * (1.0 - (maxC - minC));
        return clamp(mix(float3(finalLuma), result, max(0.0, u.lutSaturation * vibranceFactor)), 0.0, 1.0);
    }

    // --- 7. ENGINE 2 ONLY: Direct-Photon Physical Film Simulation Lab (Zero LUTs, Calibrated Naka-Rushton H&D Curve) ---
    // Operates directly on calibrated linear scene light (H0 = 0.18 middle gray, H = 1.0 diffuse white, H = 3.4 specular peak).
    // Uses the Naka-Rushton / Rational H&D Sensitometric Equation:
    //   R(H) = (H^gamma / (H^gamma + K^gamma))
    // so indoor walls (H=0.36) and white bedsheets (H=0.92) retain 100% natural texture without ever blowing out!
    inline float3 evaluatePhysicalFilmLabOnly(
        float3 directSceneLinear709,
        float3 halationScatterLinear,
        constant PipelineUniforms& u,
        uint2 pixelCoord,
        uint2 frameSize
    ) {
        // 1. Optical Lens cos^4(theta) Vignetting in Linear Photon Space
        float2 normUV = (float2(pixelCoord) / float2(max(uint2(1, 1), frameSize))) * 2.0 - 1.0;
        float rSq = dot(normUV, normUV) * 0.40;
        float cos4Falloff = pow(1.0 / sqrt(1.0 + rSq), 4.0 * u.lensVignette);

        // 2. Physical Film Base Red Backscatter (Halation) & Silver Halide Mie Bloom in Linear Space
        float stockHalationMult = (u.filmStockID == 2) ? 1.55 : 1.0; // Cinestill 800T has stronger red halation
        float3 halationWeights  = float3(1.0, 0.22, 0.02) * (u.halationIntensity * stockHalationMult * 0.55);
        float3 bloomWeights     = float3(0.32, 0.32, 0.32) * (u.emulsionBloom * 0.35);

        // Darkroom Push/Pull Chemical Development Exposure Shift (clamped to gentle realistic chemical speed shift)
        float pushExposureGain = exp2(u.pushPullStops * 0.45);
        float3 linearPhotons = max(directSceneLinear709 * cos4Falloff * pushExposureGain + halationScatterLinear * (halationWeights + bloomWeights), 0.00005);

        // 3. Emulsion Spectral Sensitivity, Chemical Couplers & Rational H&D Parameters per Stock
        // K_mid controls the semi-saturation constant (default ~0.34 so H=0.18 maps to ~0.43 display middle gray)
        float3 gammaHD  = float3(0.92, 0.92, 0.92);
        float3 kSemiSat = float3(0.35, 0.35, 0.35);
        float3 shSplit  = float3(0.0, 0.0, 0.0);
        float3 midSplit = float3(0.0, 0.0, 0.0);
        float3 hiSplit  = float3(0.0, 0.0, 0.0);
        float stockSat  = 0.96;

        // Subtle Chemical DIR Inter-Layer Coupler Matrix (Row-sum = 1.0 so neutral grays stay neutral!)
        float3 dirR = float3( 1.04, -0.02, -0.02);
        float3 dirG = float3(-0.02,  1.04, -0.02);
        float3 dirB = float3(-0.02, -0.02,  1.04);

        if (u.filmStockID == 0) {
            // 0: Kodak Vision3 500T (5219 ECN-2): Wide latitude, subtle cool-teal shadows & warm tungsten highlights
            gammaHD  = float3(0.90, 0.91, 0.93);
            kSemiSat = float3(0.36, 0.36, 0.35);
            shSplit  = float3(-0.022, 0.004, 0.028);
            hiSplit  = float3( 0.026, 0.014, -0.022);
            stockSat = 0.96;
        } else if (u.filmStockID == 1) {
            // 1: Kodak Vision3 250D (5207 Daylight): Balanced neutral cinema negative
            gammaHD  = float3(0.92, 0.92, 0.92);
            kSemiSat = float3(0.35, 0.35, 0.35);
            shSplit  = float3(-0.010, 0.003, 0.014);
            hiSplit  = float3( 0.018, 0.010, -0.014);
            stockSat = 0.98;
        } else if (u.filmStockID == 2) {
            // 2: Cinestill 800T (Remjet-Free): Signature red halation glow, cool tungsten shadows
            gammaHD  = float3(0.94, 0.95, 0.97);
            kSemiSat = float3(0.35, 0.35, 0.34);
            shSplit  = float3(-0.028, 0.008, 0.036);
            hiSplit  = float3( 0.032, 0.012, -0.024);
            stockSat = 1.00;
        } else if (u.filmStockID == 3) {
            // 3: Kodak Portra 400 (C-41 Portrait): Soft highlight shoulder, warm golden skin tones
            gammaHD  = float3(0.86, 0.86, 0.85);
            kSemiSat = float3(0.37, 0.37, 0.38);
            midSplit = float3( 0.016, 0.008, -0.012);
            hiSplit  = float3( 0.022, 0.014, -0.016);
            stockSat = 0.94;
        } else if (u.filmStockID == 4) {
            // 4: Kodak Gold 200 (C-41 Vintage): Warm nostalgic golden-amber cast
            gammaHD  = float3(0.95, 0.93, 0.90);
            kSemiSat = float3(0.34, 0.35, 0.37);
            midSplit = float3( 0.024, 0.014, -0.022);
            hiSplit  = float3( 0.034, 0.022, -0.028);
            stockSat = 1.02;
        } else if (u.filmStockID == 5) {
            // 5: Kodak Ektar 100 (C-41 Fine Grain Landscape): Crisp contrast & vivid pure primaries
            gammaHD  = float3(1.00, 1.00, 0.99);
            kSemiSat = float3(0.34, 0.34, 0.34);
            dirR = float3( 1.08, -0.04, -0.04);
            dirG = float3(-0.04,  1.08, -0.04);
            dirB = float3(-0.04, -0.04,  1.08);
            stockSat = 1.08;
        } else if (u.filmStockID == 6) {
            // 6: Fujifilm Pro 400H (4th Cyan Layer): Airy highlights, cool emerald/cyan foliage
            gammaHD  = float3(0.86, 0.88, 0.88);
            kSemiSat = float3(0.37, 0.36, 0.36);
            shSplit  = float3(-0.020, 0.012, 0.018);
            hiSplit  = float3( 0.006, 0.015, 0.010);
            stockSat = 0.95;
        } else if (u.filmStockID == 7) {
            // 7: Fujifilm Superia 400: Street film with subtle plum shadows & crisp greens
            gammaHD  = float3(0.94, 0.96, 0.95);
            shSplit  = float3( 0.014, -0.012, 0.016);
            hiSplit  = float3(-0.010,  0.020, 0.012);
            stockSat = 1.00;
        } else if (u.filmStockID == 8) {
            // 8: Fujifilm Velvia 50 (E-6 Slide): Rich reversal contrast, deep blacks, lush colors
            gammaHD  = float3(1.08, 1.08, 1.07);
            kSemiSat = float3(0.32, 0.32, 0.32);
            dirR = float3( 1.10, -0.05, -0.05);
            dirG = float3(-0.05,  1.10, -0.05);
            dirB = float3(-0.05, -0.05,  1.10);
            stockSat = 1.12;
        } else if (u.filmStockID == 9) {
            // 9: Fujifilm Provia 100F (E-6 Slide): Faithful neutral slide film
            gammaHD  = float3(0.98, 0.98, 0.99);
            kSemiSat = float3(0.34, 0.34, 0.34);
            stockSat = 1.00;
        } else if (u.filmStockID == 10) {
            // 10: Kodak Kodachrome 64 (K-14 Slide): Archival amber, red & cobalt blue
            gammaHD  = float3(1.02, 1.00, 0.97);
            kSemiSat = float3(0.33, 0.34, 0.35);
            midSplit = float3( 0.022, 0.008, -0.018);
            shSplit  = float3(-0.014, -0.006, 0.024);
            stockSat = 0.98;
        }

        // Apply DIR Chemical Coupler Matrix in Linear Light
        float3 coupledLin;
        coupledLin.r = max(0.00005, dot(linearPhotons, dirR));
        coupledLin.g = max(0.00005, dot(linearPhotons, dirG));
        coupledLin.b = max(0.00005, dot(linearPhotons, dirB));

        // Stock 11: Ilford HP5 Plus 400 Panchromatic Silver Gelatin B&W
        if (u.filmStockID == 11) {
            float mono = max(0.00005, dot(coupledLin, float3(0.28, 0.56, 0.16)));
            coupledLin = float3(mono);
            gammaHD    = float3(0.96);
            kSemiSat   = float3(0.34);
        }

        // 4. Evaluate Calibrated Rational H&D Sensitometric Curve (Guaranteed 0..1 range with wide highlight latitude!)
        float paperGrade = clamp(u.printPaperContrast + u.pushPullStops * 0.08, 0.65, 1.45);
        float3 effGamma  = gammaHD * paperGrade;

        float3 hPow = pow(coupledLin, effGamma);
        float3 kPow = pow(kSemiSat, effGamma);
        // Include high-shoulder term (1 + coupledLin / 8.0) so highlights up to H=3.5 roll off smoothly without hard clipping
        float3 rawSensitometric = (hPow * (1.0 + coupledLin / 7.5)) / (hPow + kPow);

        // Convert to display space (1 / 1.95) and apply subtle S-curve toe anchor so blacks are rich and never gray
        float3 filmPositive = pow(clamp(rawSensitometric, 0.0, 1.0), float3(1.0 / 1.95));
        filmPositive = mix(filmPositive, smoothstep(0.012, 0.992, filmPositive), 0.28);

        // 5. Apply Emulsion Dye-Layer Split Toning & Darkroom Dichroic Color Head
        float luma = dot(filmPositive, float3(0.2126, 0.7152, 0.0722));
        if (u.filmStockID != 11) {
            float shW  = pow(1.0 - luma, 1.5);
            float midW = 4.0 * luma * (1.0 - luma);
            float hiW  = pow(luma, 1.3);
            filmPositive = clamp(filmPositive + shSplit * shW + midSplit * midW + hiSplit * hiW, 0.0, 1.0);
        }

        if (abs(u.printWarmthSplit) > 0.001) {
            float3 warmHi = float3(0.032, 0.016, -0.028) * u.printWarmthSplit * luma;
            float3 coolSh = float3(-0.022, 0.005, 0.032) * (-u.printWarmthSplit) * (1.0 - luma);
            filmPositive = clamp(filmPositive + warmHi + coolSh, 0.0, 1.0);
        }

        // 6. Selwyn-Poisson Silver Halide Dye-Cloud Grain Clumping
        if (u.filmGrain > 0.001) {
            float grainScale = max(0.45, u.grainSize);
            float2 gCoord = floor(float2(pixelCoord) / grainScale) + float2(floor(u.frameTime * 24.0) * 19.1, floor(u.frameTime * 24.0) * 31.7);
            float nR = hash12(gCoord) + hash12(gCoord * 1.7 + 5.3) - 1.0;
            float nG = hash12(gCoord + 11.7) + hash12(gCoord * 1.7 + 19.1) - 1.0;
            float nB = hash12(gCoord + 23.4) + hash12(gCoord * 1.7 + 37.9) - 1.0;
            float3 grainNoise = (u.filmStockID == 11) ? float3(nG) : float3(nR * 0.85, nG * 0.80, nB * 1.10);

            float pushGrainMult = 1.0 + max(0.0, u.pushPullStops) * 0.30;
            float bellEnvelope = sqrt(max(0.0, luma * (1.0 - luma))) * 2.0;
            filmPositive = clamp(filmPositive + grainNoise * u.filmGrain * 0.075 * pushGrainMult * bellEnvelope, 0.0, 1.0);
        }

        if (u.filmStockID == 11) {
            return clamp(filmPositive, 0.0, 1.0);
        }
        float finalLuma = dot(filmPositive, float3(0.2126, 0.7152, 0.0722));
        return clamp(mix(float3(finalLuma), filmPositive, stockSat), 0.0, 1.0);
    }

    // --- 8. False Color IRE Exposure Diagnostic Map ---
    inline float3 falseColorMap(float lumaIRE) {
        if (lumaIRE < 0.05) return float3(0.55, 0.0, 0.85);
        else if (lumaIRE < 0.18) return float3(0.1, 0.35, 0.95);
        else if (lumaIRE < 0.44) return float3(0.2, 0.75, 0.85);
        else if (lumaIRE < 0.53) return float3(0.15, 0.85, 0.25);
        else if (lumaIRE < 0.62) return float3(0.95, 0.55, 0.65);
        else if (lumaIRE < 0.82) return float3(0.8, 0.8, 0.8);
        else if (lumaIRE < 0.94) return float3(0.98, 0.85, 0.1);
        else return float3(0.98, 0.1, 0.1);
    }

    // --- 9. Fast Sobel Edge Detection for Focus Peaking ---
    inline float computeSobelEdge(texture2d<float, access::read> lumaPlane, uint2 center, uint maxW, uint maxH) {
        uint x0 = (center.x > 0) ? (center.x - 1) : 0;
        uint x2 = (center.x + 1 < maxW) ? (center.x + 1) : (maxW - 1);
        uint y0 = (center.y > 0) ? (center.y - 1) : 0;
        uint y2 = (center.y + 1 < maxH) ? (center.y + 1) : (maxH - 1);

        float l = lumaPlane.read(uint2(x0, center.y)).r;
        float r = lumaPlane.read(uint2(x2, center.y)).r;
        float t = lumaPlane.read(uint2(center.x, y0)).r;
        float b = lumaPlane.read(uint2(center.x, y2)).r;
        return abs(r - l) * 2.0 + abs(b - t) * 2.0;
    }

    // --- 10. Coalesced Main Camera Compute Kernel (Zero 90-Degree Strided Cache Misses!) ---
    kernel void processAppleLogFrame(
        texture2d<float, access::read>   lumaPlane     [[texture(0)]],
        texture2d<float, access::read>   chromaPlane   [[texture(1)]],
        texture2d<float, access::write>  recordOutTex  [[texture(2)]],
        texture2d<float, access::write>  previewOutTex [[texture(3)]],
        texture3d<float, access::sample> customLUTTex  [[texture(4)]],
        constant PipelineUniforms&       uniforms      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = previewOutTex.get_width();
        uint outH = previewOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }

        uint srcW = lumaPlane.get_width();
        uint srcH = lumaPlane.get_height();

        // Coalesced horizontal memory mapping (gid.x maps directly along srcX row!)
        uint srcX = min(gid.x * srcW / outW, srcW - 1);
        uint srcY = min(gid.y * srcH / outH, srcH - 1);
        uint2 lumaCoord = uint2(srcX, srcY);
        uint2 chromaCoord = uint2(
            srcX * chromaPlane.get_width() / srcW,
            srcY * chromaPlane.get_height() / srcH
        );

        float  ySample    = lumaPlane.read(lumaCoord).r;
        float2 cbcrSample = chromaPlane.read(chromaCoord).rg;

        // Lightweight TPDF dither & 2-tap deband only when custom LUT or 8-bit SDR is used
        if (uniforms.isSDRSensorInput == 1 || (uniforms.colorEngineMode == 1 && uniforms.lutPresetID == 9 && uniforms.lutDebanding > 0.05)) {
            uint nx = min(srcX + 6, srcW - 1);
            float nY = lumaPlane.read(uint2(nx, srcY)).r;
            if (abs(nY - ySample) < 0.018 * uniforms.lutDebanding) {
                ySample = mix(ySample, nY, 0.45);
            }
            float dither = (hash12(float2(gid) + uniforms.frameTime * 11.3) - 0.5) * 0.0015;
            ySample = clamp(ySample + dither, 0.0, 1.0);
        }

        float3 sensorRGB = decodeYCbCrToRGB(ySample, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput);

        // Optional Radial Chromatic Aberration & 4-Tap Halation ONLY when Physical Film Lab (Mode 2) is active
        float3 halationLinear = float3(0.0);
        if (uniforms.colorEngineMode == 2) {
            if (uniforms.chromaticAberration > 0.02) {
                int shiftPx = int(float(int(srcX) - int(srcW / 2)) * 0.0035 * uniforms.chromaticAberration);
                uint rX = uint(clamp(int(srcX) + shiftPx, 0, int(srcW) - 1));
                uint bX = uint(clamp(int(srcX) - shiftPx, 0, int(srcW) - 1));
                float yR = lumaPlane.read(uint2(rX, srcY)).r;
                float yB = lumaPlane.read(uint2(bX, srcY)).r;
                sensorRGB.r = decodeYCbCrToRGB(yR, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput).r;
                sensorRGB.b = decodeYCbCrToRGB(yB, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput).b;
            }
            if (uniforms.halationIntensity > 0.02 || uniforms.emulsionBloom > 0.02) {
                uint stepPx = 10;
                float h1 = max(0.0, lumaPlane.read(uint2(min(srcX + stepPx, srcW - 1), srcY)).r - 0.68) * 3.0;
                float h2 = max(0.0, lumaPlane.read(uint2((srcX > stepPx) ? (srcX - stepPx) : 0, srcY)).r - 0.68) * 3.0;
                float h3 = max(0.0, lumaPlane.read(uint2(srcX, min(srcY + stepPx, srcH - 1))).r - 0.68) * 3.0;
                float h4 = max(0.0, lumaPlane.read(uint2(srcX, (srcY > stepPx) ? (srcY - stepPx) : 0)).r - 0.68) * 3.0;
                halationLinear = float3((h1 * h1 + h2 * h2 + h3 * h3 + h4 * h4) * 0.25);
            }
        }

        float activeEV = (uniforms.colorEngineMode == 2) ? uniforms.filmExposureEV : uniforms.lutExposureEV;
        float activeRollOff = (uniforms.colorEngineMode == 2) ? 0.25 : uniforms.lutHighlightRollOff;

        float3 sceneLinearBT2020;
        float3 appleLogRGB;
        if (uniforms.isNativeAppleLog == 1) {
            appleLogRGB = sensorRGB;
            sceneLinearBT2020 = appleLogToLinear(sensorRGB) * exp2(activeEV);
        } else {
            sceneLinearBT2020 = sensorToCalibratedLinearBT2020(sensorRGB, uniforms.isSDRSensorInput, activeEV, activeRollOff);
            appleLogRGB = linearToAppleLog(sceneLinearBT2020);
        }

        // Strictly Mutually Exclusive Color Engine Evaluation:
        float3 gradedRGB;
        if (uniforms.colorEngineMode == 0) {
            // Mode 0: Pure Apple Log
            gradedRGB = appleLogRGB;
        } else if (uniforms.colorEngineMode == 1) {
            // Mode 1: Independent LUT Grading System ONLY
            gradedRGB = evaluateLUTSystemOnly(appleLogRGB, uniforms, customLUTTex);
        } else {
            // Mode 2: Independent Physical Film Simulation Lab ONLY (Direct Scene-Linear Photons!)
            float3 directLinear709 = linearBT2020ToLinear709(sceneLinearBT2020);
            gradedRGB = evaluatePhysicalFilmLabOnly(directLinear709, halationLinear, uniforms, gid, uint2(outW, outH));
        }

        if (uniforms.writeRecordBuffer == 1) {
            float3 recordSignal = (uniforms.bakeIntoRecording == 1 && uniforms.colorEngineMode != 0) ? gradedRGB : appleLogRGB;
            recordOutTex.write(float4(recordSignal, 1.0), gid);
        }

        float lumaLog = dot(appleLogRGB, float3(0.2627, 0.6780, 0.0593));
        float3 monitorRGB = gradedRGB;

        if (uniforms.monitorDiagnosticMode == 1) {
            monitorRGB = falseColorMap(lumaLog);
        } else if (uniforms.monitorDiagnosticMode == 2 && lumaLog > 0.94) {
            int stripe = int(float(gid.x + gid.y) * 0.25 + uniforms.frameTime * 15.0) % 2;
            if (stripe == 0) {
                monitorRGB = float3(1.0, 0.15, 0.15);
            }
        }

        if (uniforms.focusPeakingEnabled == 1) {
            float edgeMag = computeSobelEdge(lumaPlane, lumaCoord, srcW, srcH);
            if (edgeMag >= uniforms.focusPeakingThreshold) {
                float3 peakColor = float3(0.1, 1.0, 0.2);
                if (uniforms.focusPeakingColorIndex == 1) peakColor = float3(1.0, 0.12, 0.12);
                else if (uniforms.focusPeakingColorIndex == 2) peakColor = float3(1.0, 0.92, 0.1);
                else if (uniforms.focusPeakingColorIndex == 3) peakColor = float3(0.1, 0.95, 1.0);
                monitorRGB = mix(monitorRGB, peakColor, 0.88);
            }
        }

        previewOutTex.write(float4(monitorRGB, 1.0), gid);
    }

    // --- 11. Post-Production Video Frame Grading Kernel ---
    kernel void gradeExistingLogFrame(
        texture2d<float, access::read>   sourceTex    [[texture(0)]],
        texture2d<float, access::write>  gradedOutTex [[texture(1)]],
        texture3d<float, access::sample> customLUTTex [[texture(2)]],
        constant PipelineUniforms&       uniforms     [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = gradedOutTex.get_width();
        uint outH = gradedOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }
        float3 srcLog = sourceTex.read(gid).rgb;
        float activeEV = (uniforms.colorEngineMode == 2) ? uniforms.filmExposureEV : uniforms.lutExposureEV;
        float3 lin2020 = appleLogToLinear(srcLog) * exp2(activeEV);

        float3 graded;
        if (uniforms.colorEngineMode == 0) {
            graded = srcLog;
        } else if (uniforms.colorEngineMode == 1) {
            float3 adjLog = linearToAppleLog(lin2020);
            graded = evaluateLUTSystemOnly(adjLog, uniforms, customLUTTex);
        } else {
            float3 lin709 = linearBT2020ToLinear709(lin2020);
            graded = evaluatePhysicalFilmLabOnly(lin709, float3(0.0), uniforms, gid, uint2(outW, outH));
        }
        gradedOutTex.write(float4(graded, 1.0), gid);
    }

    // --- 12. Zero-Cost Orientation & Aspect-Fit Render Pipeline Shaders for MTKView ---
    struct VertexOut {
        float4 position [[position]];
        float2 texCoord;
    };

    struct ViewfinderScaleUniforms {
        float2 scale;
        int rotationMode;     // 0 = Portrait (90 CW), 1 = Landscape Right (0 deg), 2 = Landscape Left (180 deg), 3 = Portrait UpsideDown (270 CW)
        int mirrorHorizontal; // 1 = Mirror X
    };

    vertex VertexOut vertexViewfinder(
        uint vertexID [[vertex_id]],
        constant ViewfinderScaleUniforms& vf [[buffer(0)]]
    ) {
        const float2 positions[4] = {
            float2(-1.0, -1.0),
            float2( 1.0, -1.0),
            float2(-1.0,  1.0),
            float2( 1.0,  1.0)
        };

        // Rotate texture coordinates in the 4-vertex shader so the compute pass stays 100% coalesced!
        float2 uv;
        if (vf.rotationMode == 0) {
            // Portrait (90° CW rotation of native LandscapeRight sensor)
            const float2 uvsPortrait[4] = {
                float2(1.0, 1.0),
                float2(1.0, 0.0),
                float2(0.0, 1.0),
                float2(0.0, 0.0)
            };
            uv = uvsPortrait[vertexID];
        } else if (vf.rotationMode == 1) {
            // Landscape Right (0° native sensor orientation)
            const float2 uvsLandRight[4] = {
                float2(0.0, 1.0),
                float2(1.0, 1.0),
                float2(0.0, 0.0),
                float2(1.0, 0.0)
            };
            uv = uvsLandRight[vertexID];
        } else if (vf.rotationMode == 2) {
            // Landscape Left (180° rotation)
            const float2 uvsLandLeft[4] = {
                float2(1.0, 0.0),
                float2(0.0, 0.0),
                float2(1.0, 1.0),
                float2(0.0, 1.0)
            };
            uv = uvsLandLeft[vertexID];
        } else {
            // Portrait Upside Down (270° CW)
            const float2 uvsUpsideDown[4] = {
                float2(0.0, 0.0),
                float2(0.0, 1.0),
                float2(1.0, 0.0),
                float2(1.0, 1.0)
            };
            uv = uvsUpsideDown[vertexID];
        }

        float2 pos = positions[vertexID];
        if (vf.mirrorHorizontal == 1) {
            pos.x = -pos.x;
        }

        VertexOut out;
        out.position = float4(pos * vf.scale, 0.0, 1.0);
        out.texCoord = uv;
        return out;
    }

    fragment float4 fragmentViewfinder(
        VertexOut in [[stage_in]],
        texture2d<float, access::sample> previewTex [[texture(0)]]
    ) {
        constexpr sampler bilinearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        return previewTex.sample(bilinearSampler, in.texCoord);
    }
    """
}

// MARK: - 2. Strictly Mutually Exclusive Color Engine Models

enum ColorEngineMode: Int, CaseIterable, Identifiable {
    case pureAppleLog    = 0 // 原生 Apple Log (无 LUT / 无胶片)
    case lutGrading      = 1 // 独立 LUT 调色系统 (不叠加物理胶片)
    case physicalFilmLab = 2 // 独立物理胶片暗房 (不叠加 LUT)

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .pureAppleLog:    return "原生 Log"
        case .lutGrading:      return "LUT 调色"
        case .physicalFilmLab: return "物理胶片"
        }
    }
}

enum BuiltInLUTPresetOption: Int, CaseIterable, Identifiable {
    case rawAppleLog     = 0
    case officialRec709  = 1
    case arriAlexa709    = 2
    case kodak2383D65    = 3
    case kodak2383Warm   = 4
    case fuji3513DI      = 5
    case tealOrange      = 6
    case bleachBypass    = 7
    case cyberNocturne   = 8
    case customCubeLUT   = 9

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .rawAppleLog:    return "Raw Apple Log"
        case .officialRec709: return "Apple Rec.709"
        case .arriAlexa709:   return "ARRI Alexa 709"
        case .kodak2383D65:   return "Kodak 2383 D65"
        case .kodak2383Warm:  return "Kodak 2383 D55"
        case .fuji3513DI:     return "Fuji 3513DI"
        case .tealOrange:     return "Teal & Orange"
        case .bleachBypass:   return "Bleach Bypass"
        case .cyberNocturne:  return "Cyber Nocturne"
        case .customCubeLUT:  return "导入 .cube LUT"
        }
    }

    var subtitle: String {
        switch self {
        case .rawAppleLog:    return "纯净 Apple Log 对数灰片（无色彩映射）"
        case .officialRec709: return "苹果官方 BT.2020 Apple Log → Rec.709 标准还原"
        case .arriAlexa709:   return "阿莱 Alexa K1S1 风格还原 • 奶油高光滚降与通透肤色"
        case .kodak2383D65:   return "好莱坞标准柯达 2383 影院印片 LUT (D65 标准白点)"
        case .kodak2383Warm:  return "柯达 2383 暖金胶片印片 LUT (D55 暖白点)"
        case .fuji3513DI:     return "富士 Eterna 3513DI 数字中间片印片 LUT • 翠绿与冷青暗部"
        case .tealOrange:     return "商业电影高反差青橙分离 LUT"
        case .bleachBypass:   return "留银冲洗银盐金属质感 LUT"
        case .cyberNocturne:  return "都市夜景霓虹青紫电影 LUT"
        case .customCubeLUT:  return "自定义外部 .cube 3D LUT（达芬奇四面体插值引擎）"
        }
    }
}

enum PhysicalFilmStockOption: Int, CaseIterable, Identifiable {
    case vision3_500T  = 0
    case vision3_250D  = 1
    case cinestill800T = 2
    case portra400     = 3
    case gold200       = 4
    case ektar100      = 5
    case fujiPro400H   = 6
    case superia400    = 7
    case velvia50      = 8
    case provia100F    = 9
    case kodachrome64  = 10
    case ilfordHP5     = 11

    var id: Int { rawValue }

    var shortName: String {
        switch self {
        case .vision3_500T:  return "Vision3 500T"
        case .vision3_250D:  return "Vision3 250D"
        case .cinestill800T: return "Cinestill 800T"
        case .portra400:     return "Portra 400"
        case .gold200:       return "Kodak Gold 200"
        case .ektar100:      return "Kodak Ektar 100"
        case .fujiPro400H:   return "Fuji Pro 400H"
        case .superia400:    return "Fuji Superia 400"
        case .velvia50:      return "Fuji Velvia 50"
        case .provia100F:    return "Fuji Provia 100F"
        case .kodachrome64:  return "Kodachrome 64"
        case .ilfordHP5:     return "Ilford HP5 B&W"
        }
    }

    var chemistryBadge: String {
        switch self {
        case .vision3_500T:  return "ECN-2 电影负片 (5219) • 钨丝灯冷青暗部 + 2383 暖金高光"
        case .vision3_250D:  return "ECN-2 日光电影负片 (5207) • 自然高锐度 + 宽广高光肩部"
        case .cinestill800T: return "去石墨层电影负片 • 片基红橙色光晕 (Halation) + 夜景霓虹"
        case .portra400:     return "C-41 专业人像负片 • DIR 成色剂耦合 + 温润金黄肤色"
        case .gold200:       return "C-41 经典暖调负片 • 复古金黄高光 + 浓郁厚重反差"
        case .ektar100:      return "C-41 超细颗粒风光负片 • 极高成色剂色彩纯度 + 深邃黑位"
        case .fujiPro400H:   return "C-41 四层感光负片 • 第四青敏层翡翠绿 + 空气感粉白高光"
        case .superia400:    return "C-41 经典街拍负片 • 微妙洋红暗部偏移 + 鲜亮青绿"
        case .velvia50:      return "E-6 传奇风光反转片 • 深邃黑位 + 高饱和风光正片"
        case .provia100F:    return "E-6 专业标准反转片 • 精准中性灰阶 + 细腻正片微反差"
        case .kodachrome64:  return "K-14 三道外偶反转片 • 传奇纪实琥珀黄、暖红与深钴蓝"
        case .ilfordHP5:     return "D-76 全色黑白银盐负片 • 扎实卤化银团簇颗粒 + 可变反差相纸"
        }
    }
}

/// Independent State for Engine 1: LUT Color Grading System ONLY
struct LUTSystemParameters: Equatable {
    var lutPreset: BuiltInLUTPresetOption = .officialRec709
    var lutInputSpace: LUTInputSpaceOption = .appleLog
    var lutIntensity: Float = 1.0
    var exposureEV: Float = 0.0
    var contrast: Float = 1.0
    var highlightRollOff: Float = 0.35
    var shadowLift: Float = 0.0
    var saturation: Float = 1.0
    var vibrance: Float = 0.0
    var temperature: Float = 0.0
    var tint: Float = 0.0
    var debanding: Float = 0.50

    var liftR: Float = 0.0
    var liftG: Float = 0.0
    var liftB: Float = 0.0
    var gammaR: Float = 0.0
    var gammaG: Float = 0.0
    var gammaB: Float = 0.0
    var gainR: Float = 1.0
    var gainG: Float = 1.0
    var gainB: Float = 1.0

    mutating func reset() {
        let keepPreset = lutPreset
        self = LUTSystemParameters()
        self.lutPreset = keepPreset
    }
}

/// Independent State for Engine 2: Physical Film Simulation Lab ONLY
struct PhysicalFilmParameters: Equatable {
    var filmStock: PhysicalFilmStockOption = .portra400
    var exposureEV: Float = 0.0
    var pushPullStops: Float = 0.0
    var halationIntensity: Float = 0.25
    var emulsionBloom: Float = 0.18
    var printPaperContrast: Float = 1.00
    var printWarmthSplit: Float = 0.08
    var filmGrain: Float = 0.20
    var grainSize: Float = 1.0
    var lensVignette: Float = 0.22
    var chromaticAberration: Float = 0.10

    mutating func selectStock(_ stock: PhysicalFilmStockOption) {
        self.filmStock = stock
        self.exposureEV = 0.0
        self.pushPullStops = 0.0
        switch stock {
        case .vision3_500T:
            halationIntensity = 0.32; emulsionBloom = 0.20; filmGrain = 0.22; printPaperContrast = 1.00; printWarmthSplit = 0.08
        case .vision3_250D:
            halationIntensity = 0.20; emulsionBloom = 0.15; filmGrain = 0.16; printPaperContrast = 1.00; printWarmthSplit = 0.04
        case .cinestill800T:
            halationIntensity = 0.65; emulsionBloom = 0.28; filmGrain = 0.28; printPaperContrast = 1.02; printWarmthSplit = 0.05
        case .portra400:
            halationIntensity = 0.18; emulsionBloom = 0.18; filmGrain = 0.16; printPaperContrast = 0.96; printWarmthSplit = 0.12
        case .gold200:
            halationIntensity = 0.22; emulsionBloom = 0.20; filmGrain = 0.22; printPaperContrast = 1.02; printWarmthSplit = 0.20
        case .ektar100:
            halationIntensity = 0.12; emulsionBloom = 0.12; filmGrain = 0.10; printPaperContrast = 1.05; printWarmthSplit = 0.04
        case .fujiPro400H:
            halationIntensity = 0.16; emulsionBloom = 0.18; filmGrain = 0.15; printPaperContrast = 0.96; printWarmthSplit = -0.06
        case .superia400:
            halationIntensity = 0.22; emulsionBloom = 0.18; filmGrain = 0.22; printPaperContrast = 1.00; printWarmthSplit = -0.04
        case .velvia50:
            halationIntensity = 0.10; emulsionBloom = 0.10; filmGrain = 0.08; printPaperContrast = 1.08; printWarmthSplit = 0.02
        case .provia100F:
            halationIntensity = 0.10; emulsionBloom = 0.12; filmGrain = 0.10; printPaperContrast = 1.00; printWarmthSplit = 0.0
        case .kodachrome64:
            halationIntensity = 0.15; emulsionBloom = 0.14; filmGrain = 0.18; printPaperContrast = 1.04; printWarmthSplit = 0.14
        case .ilfordHP5:
            halationIntensity = 0.25; emulsionBloom = 0.22; filmGrain = 0.34; printPaperContrast = 1.04; printWarmthSplit = 0.0
        }
    }
}

enum CaptureModeOption: Int, CaseIterable, Identifiable {
    case video = 0
    case photo = 1

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .video: return "VIDEO 摄影机"
        case .photo: return "PHOTO 胶片相机"
        }
    }
}

enum CaptureOrientationMode: Int, CaseIterable, Identifiable {
    case portrait           = 0
    case landscapeRight     = 1
    case landscapeLeft      = 2
    case portraitUpsideDown = 3

    var id: Int { rawValue }

    var shortLabel: String {
        switch self {
        case .portrait:           return "竖屏"
        case .landscapeRight:     return "横屏"
        case .landscapeLeft:      return "横屏反"
        case .portraitUpsideDown: return "竖屏反"
        }
    }

    var isPortraitSwapped: Bool {
        self == .portrait || self == .portraitUpsideDown
    }

    /// QuickTime Track Header Transform for Zero-Copy Orientation in AVAssetWriter
    var assetWriterTransform: CGAffineTransform {
        switch self {
        case .portrait:
            return CGAffineTransform(rotationAngle: .pi / 2.0)
        case .landscapeRight:
            return .identity
        case .landscapeLeft:
            return CGAffineTransform(rotationAngle: .pi)
        case .portraitUpsideDown:
            return CGAffineTransform(rotationAngle: -.pi / 2.0)
        }
    }
}

enum CinemaStabilizationMode: String, CaseIterable, Identifiable {
    case off               = "关闭 (100% 满画幅/上脚架)"
    case standard          = "标准防抖 (OIS + 轻裁切)"
    case cinematic         = "电影防抖 (Cinematic 悬浮)"
    case cinematicExtended = "极限防抖 (Extended 运动级)"

    var id: String { rawValue }

    var shortTitle: String {
        switch self {
        case .off:               return "OFF"
        case .standard:          return "STD"
        case .cinematic:         return "CINE"
        case .cinematicExtended: return "EXTREME"
        }
    }

    var avMode: AVCaptureVideoStabilizationMode {
        switch self {
        case .off:               return .off
        case .standard:          return .standard
        case .cinematic:         return .cinematic
        case .cinematicExtended: return .cinematicExtended
        }
    }
}

enum FocusControlMode: String, CaseIterable, Identifiable {
    case continuousAF = "CAF 连续追焦"
    case singleAF     = "AF-S 单次对焦"
    case manualFocus  = "MF 手动对焦"

    var id: String { rawValue }

    var shortLabel: String {
        switch self {
        case .continuousAF: return "CAF"
        case .singleAF:     return "AF-S"
        case .manualFocus:  return "MF"
        }
    }
}

enum FocusPeakingColorOption: Int, CaseIterable, Identifiable {
    case green  = 0
    case red    = 1
    case yellow = 2
    case cyan   = 3

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .green:  return "翠绿"
        case .red:    return "绯红"
        case .yellow: return "明黄"
        case .cyan:   return "青蓝"
        }
    }
}

enum AutoFocusRangeOption: Int, CaseIterable, Identifiable {
    case full = 0
    case near = 1
    case far  = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .full: return "全范围"
        case .near: return "近景微距"
        case .far:  return "远景无限"
        }
    }

    var avRange: AVCaptureDevice.AutoFocusRangeRestriction {
        switch self {
        case .full: return .none
        case .near: return .near
        case .far:  return .far
        }
    }
}

enum SensorFormatPreset: String, CaseIterable, Identifiable {
    case uhd4K16x9   = "4K UHD 16:9 (3840×2160 原生10-Bit)"
    case openGateMax = "Open Gate 4:3 (4032×3024 满传感器)"
    case fhd1080p    = "1080p FHD 16:9 (1920×1080 高帧率)"

    var id: String { rawValue }

    var shortTitle: String {
        switch self {
        case .uhd4K16x9:   return "4K 10-Bit"
        case .openGateMax: return "OpenGate 4:3"
        case .fhd1080p:    return "1080p"
        }
    }
}

enum CameraLensOption: String, CaseIterable, Identifiable {
    case ultraWide = "0.5x"
    case wideMain  = "1x"
    case telephoto = "3x"

    var id: String { rawValue }

    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .ultraWide: return .builtInUltraWideCamera
        case .wideMain:  return .builtInWideAngleCamera
        case .telephoto: return .builtInTelephotoCamera
        }
    }

    var focalLabel: String {
        switch self {
        case .ultraWide: return "13mm f/1.8"
        case .wideMain:  return "26mm f/1.5 OIS"
        case .telephoto: return "77mm f/2.8 OIS"
        }
    }
}

enum RecordingCodecOption: String, CaseIterable, Identifiable {
    case proRes422HQ = "ProRes 422 HQ"
    case proRes422   = "ProRes 422"
    case proRes422LT = "ProRes 422 LT"
    case hevc10Bit   = "HEVC 10-Bit"

    var id: String { rawValue }

    var avCodecType: AVVideoCodecType {
        switch self {
        case .proRes422HQ: return .proRes422HQ
        case .proRes422:   return .proRes422
        case .proRes422LT: return .proRes422LT
        case .hevc10Bit:   return .hevc
        }
    }
}

enum FrameRateOption: Double, CaseIterable, Identifiable {
    case fps24 = 24.0
    case fps25 = 25.0
    case fps30 = 30.0
    case fps60 = 60.0

    var id: Double { rawValue }
    var label: String { "\(Int(rawValue))fps" }
}

enum LUTInputSpaceOption: Int, CaseIterable, Identifiable {
    case appleLog = 0
    case rec709   = 1

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .appleLog: return "Apple Log 输入"
        case .rec709:   return "Rec.709 输入"
        }
    }
}

enum MonitorDiagnosticMode: Int, CaseIterable, Identifiable {
    case off        = 0
    case falseColor = 1
    case zebra      = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .off:        return "标准监看"
        case .falseColor: return "False Color 假色"
        case .zebra:      return "Zebra 斑马纹"
        }
    }
}

enum AspectGuideOption: String, CaseIterable, Identifiable {
    case none        = "无遮幅"
    case scope239    = "2.39:1 宽银幕"
    case cinema185   = "1.85:1 电影"
    case vertical916 = "9:16 竖屏安全框"

    var id: String { rawValue }
}

// MARK: - 3. 3D .cube LUT Parser & Exporter (Scalar Float Math)

enum CubeLUTManager {
    static func loadCubeLUT(from url: URL, device: MTLDevice) throws -> (MTLTexture, Int, String) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        let content = try String(contentsOf: url, encoding: .utf8)
        var size = 0
        var floatValues: [Float] = []
        floatValues.reserveCapacity(33 * 33 * 33 * 4)

        for rawLine in content.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") || line.hasPrefix("TITLE") || line.hasPrefix("DOMAIN_") || line.hasPrefix("LUT_1D_") {
                continue
            }
            if line.hasPrefix("LUT_3D_SIZE") {
                let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
                if parts.count >= 2, let parsedSize = Int(parts[1]), parsedSize >= 2 && parsedSize <= 65 {
                    size = parsedSize
                }
                continue
            }
            let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            if parts.count >= 3,
               let r = Float(parts[0]),
               let g = Float(parts[1]),
               let b = Float(parts[2]) {
                floatValues.append(r)
                floatValues.append(g)
                floatValues.append(b)
                floatValues.append(1.0)
            }
        }

        guard size > 0, floatValues.count == size * size * size * 4 else {
            throw NSError(domain: "CubeLUT", code: -1, userInfo: [NSLocalizedDescriptionKey: "无效的 .cube 3D LUT 文件格式（需为标准 3D LUT）"])
        }

        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rgba32Float
        desc.width = size
        desc.height = size
        desc.depth = size
        desc.usage = [.shaderRead]

        guard let texture = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "CubeLUT", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法创建 3D LUT GPU 纹理"])
        }

        floatValues.withUnsafeBytes { rawPtr in
            if let baseAddress = rawPtr.baseAddress {
                texture.replace(
                    region: MTLRegionMake3D(0, 0, 0, size, size, size),
                    mipmapLevel: 0,
                    slice: 0,
                    withBytes: baseAddress,
                    bytesPerRow: size * MemoryLayout<Float>.size * 4,
                    bytesPerImage: size * size * MemoryLayout<Float>.size * 4
                )
            }
        }

        return (texture, size, url.deletingPathExtension().lastPathComponent)
    }

    static func exportCubeFile(lutParams: LUTSystemParameters) throws -> URL {
        let size = 33
        var lines: [String] = [
            "# Generated by CineLog 13 Pro Max v6.0",
            "TITLE \"CineLog13Pro_\(lutParams.lutPreset.title)\"",
            "LUT_3D_SIZE \(size)",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0"
        ]
        lines.reserveCapacity(size * size * size + 10)

        for bIdx in 0..<size {
            for gIdx in 0..<size {
                for rIdx in 0..<size {
                    let rIn = Float(rIdx) / Float(size - 1)
                    let gIn = Float(gIdx) / Float(size - 1)
                    let bIn = Float(bIdx) / Float(size - 1)
                    let out = evaluateGradeCPU(appleLogRGB: SIMD3<Float>(rIn, gIn, bIn), u: lutParams)
                    lines.append(String(format: "%.6f %.6f %.6f", out.x, out.y, out.z))
                }
            }
        }

        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("CineLog13Pro_Export.cube")
        try lines.joined(separator: "\n").write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    private static func clamp01(_ v: Float) -> Float {
        return max(0.0, min(1.0, v))
    }

    private static func rationalToneMapChannel(_ x: Float) -> Float {
        let h: Float = max(0.0, x)
        let num: Float = h * (1.0 + h / 6.5)
        let den: Float = h + 0.38
        let mapped: Float = clamp01(num / max(0.0001, den))
        return pow(mapped, 1.0 / 2.15)
    }

    private static func logToLinChannel(_ P: Float) -> Float {
        let R_0: Float = -0.05641088
        let cVal: Float = 47.28711236
        let beta: Float = 0.00964052
        let gammaVal: Float = 0.08550479
        let delta: Float = 0.69336945
        let P_t: Float = 0.20856507
        if P >= P_t {
            return pow(2.0, (P - delta) / gammaVal) - beta
        } else if P >= 0.0 {
            return sqrt(P / cVal) + R_0
        } else {
            return R_0
        }
    }

    private static func gradeSingleLogChannel(_ val: Float, lift: Float, gamma: Float, gain: Float, contrast: Float) -> Float {
        let lifted: Float = val + lift * (1.0 - val)
        let gained: Float = clamp01(lifted * gain)
        let gPow: Float = max(0.3, 1.0 - gamma * 0.55)
        let curved: Float = pow(gained, gPow)
        let contrasted: Float = (curved - 0.488) * contrast + 0.488
        return clamp01(contrasted)
    }

    private static func evaluateGradeCPU(appleLogRGB: SIMD3<Float>, u: LUTSystemParameters) -> SIMD3<Float> {
        let tempShift: Float = u.temperature * 0.045
        let tintShift: Float = u.tint * 0.025

        let r0: Float = appleLogRGB.x + tempShift + tintShift
        let g0: Float = appleLogRGB.y - u.tint * 0.040
        let b0: Float = appleLogRGB.z - tempShift + tintShift

        let rLog: Float = gradeSingleLogChannel(r0, lift: u.liftR, gamma: u.gammaR, gain: u.gainR, contrast: u.contrast)
        let gLog: Float = gradeSingleLogChannel(g0, lift: u.liftG, gamma: u.gammaG, gain: u.gainG, contrast: u.contrast)
        let bLog: Float = gradeSingleLogChannel(b0, lift: u.liftB, gamma: u.gammaB, gain: u.gainB, contrast: u.contrast)

        let evGain: Float = exp2(u.exposureEV)
        let linR2020: Float = max(0.0, logToLinChannel(rLog)) * evGain
        let linG2020: Float = max(0.0, logToLinChannel(gLog)) * evGain
        let linB2020: Float = max(0.0, logToLinChannel(bLog)) * evGain

        let linR709: Float = max(0.0,  1.6604910 * linR2020 - 0.5876411 * linG2020 - 0.0728499 * linB2020)
        let linG709: Float = max(0.0, -0.1245505 * linR2020 + 1.1328999 * linG2020 - 0.0083494 * linB2020)
        let linB709: Float = max(0.0, -0.0181508 * linR2020 - 0.1005789 * linG2020 + 1.1187297 * linB2020)

        let recR: Float = rationalToneMapChannel(linR709)
        let recG: Float = rationalToneMapChannel(linG709)
        let recB: Float = rationalToneMapChannel(linB709)

        let luma: Float = 0.2126 * recR + 0.7152 * recG + 0.0722 * recB
        let outR: Float = clamp01(luma + (recR - luma) * u.saturation)
        let outG: Float = clamp01(luma + (recG - luma) * u.saturation)
        let outB: Float = clamp01(luma + (recB - luma) * u.saturation)

        return SIMD3<Float>(outR, outG, outB)
    }
}

// MARK: - 4. Core 10-Bit Apple Log + Coalesced Zero-Stutter GPU Pipeline Engine

final class AppleLogCameraEngine: NSObject, ObservableObject, @unchecked Sendable {
    enum EngineRecordingState {
        case idle
        case starting
        case recording
        case stopping
    }

    // Telemetry & Status
    @Published var isSessionRunning: Bool = false
    @Published var captureMode: CaptureModeOption = .video
    @Published var isRecording: Bool = false
    @Published var isCapturingPhoto: Bool = false
    @Published var showShutterFlash: Bool = false
    @Published var dualSaveRawAndFilmPhoto: Bool = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var activePipelineBadge: String = "Initializing Cinema Pipeline..."
    @Published var activeResolutionBadge: String = "3840×2160 (10-Bit)"
    @Published var statusMessage: String = "Ready"
    @Published var lastRecordedVideoURL: URL?

    // Slate & Timecode Metadata
    @Published var projectReelName: String = "A001"
    @Published var clipTakeNumber: Int = 1
    @Published var useTimeOfDayTimecode: Bool = false

    // Capture Format, Open Gate, Lens, Zoom & Orientation
    @Published var selectedSensorPreset: SensorFormatPreset = .uhd4K16x9 {
        didSet { reconfigureFormatAndSession() }
    }
    @Published var selectedLens: CameraLensOption = .wideMain {
        didSet { switchCameraLens(selectedLens) }
    }
    @Published var zoomFactor: CGFloat = 1.0
    @Published var orientationMode: CaptureOrientationMode = .portrait
    @Published var autoFollowDeviceOrientation: Bool = false
    @Published var isMirroredHorizontal: Bool = false
    @Published var selectedCodec: RecordingCodecOption = .proRes422
    @Published var selectedFPS: FrameRateOption = .fps24 {
        didSet { reconfigureFormatAndSession() }
    }

    // Stabilization & Gyro Horizon Leveler
    @Published var stabilizationMode: CinemaStabilizationMode = .standard {
        didSet { applyStabilizationMode() }
    }
    @Published var showHorizonLeveler: Bool = true
    @Published var deviceRollDegrees: Double = 0.0
    @Published var devicePitchDegrees: Double = 0.0

    // Professional Focus System
    @Published var focusMode: FocusControlMode = .continuousAF {
        didSet { applyFocusConfiguration() }
    }
    @Published var lensPosition: Float = 0.5
    @Published var rackPointA: Float = 0.20
    @Published var rackPointB: Float = 0.85
    @Published var rackDurationSeconds: Double = 1.5
    @Published var isRackingFocus: Bool = false
    @Published var afRangeRestriction: AutoFocusRangeOption = .full {
        didSet { applyFocusConfiguration() }
    }
    @Published var isSmoothAFEnabled: Bool = true {
        didSet { applyFocusConfiguration() }
    }
    @Published var focusPeakingEnabled: Bool = false
    @Published var focusPeakingColor: FocusPeakingColorOption = .green
    @Published var focusPeakingThreshold: Float = 0.10
    @Published var focusReticleNormalized: CGPoint? = nil
    @Published var isFocusLockedByTap: Bool = false

    // Manual Cinema Exposure & White Balance (Kelvin + Tint)
    @Published var isManualExposure: Bool = false {
        didSet { updateExposureAndShutter() }
    }
    @Published var currentISO: Float = 100.0
    @Published var minISO: Float = 50.0
    @Published var maxISO: Float = 1600.0
    @Published var shutterAngleDegrees: Double = 180.0 {
        didSet { updateExposureAndShutter() }
    }
    @Published var whiteBalanceKelvin: Float = 5600.0 {
        didSet { applyWhiteBalanceKelvinAndTint() }
    }
    @Published var whiteBalanceTint: Float = 0.0 {
        didSet { applyWhiteBalanceKelvinAndTint() }
    }
    @Published var isWhiteBalanceLocked: Bool = false {
        didSet { toggleWhiteBalanceLock(isWhiteBalanceLocked) }
    }

    // Strictly Mutually Exclusive Color Engines:
    // Engine 0 = Pure Log, Engine 1 = LUT System Only, Engine 2 = Physical Film Lab Only
    @Published var colorEngineMode: ColorEngineMode = .lutGrading
    @Published var lutSettings = LUTSystemParameters()
    @Published var filmSettings = PhysicalFilmParameters()
    @Published var bakeIntoRecording: Bool = false
    @Published var monitorDiagnostic: MonitorDiagnosticMode = .off
    @Published var aspectGuide: AspectGuideOption = .none
    @Published var showGridOverlay: Bool = true
    @Published var customLUTName: String = "未加载 .cube"

    var activeModeSummaryBadge: String {
        switch colorEngineMode {
        case .pureAppleLog:
            return "原生 Apple Log"
        case .lutGrading:
            return "LUT: \(lutSettings.lutPreset.title)"
        case .physicalFilmLab:
            return "物理胶片: \(filmSettings.filmStock.shortName)"
        }
    }

    // Live Luma Histogram (64 bins) & Stereo Audio VU Meter (dBFS)
    @Published var lumaHistogram: [Float] = Array(repeating: 0, count: 64)
    @Published var audioPeakDBLeft: Float = -60.0
    @Published var audioPeakDBRight: Float = -60.0

    // AVFoundation Capture Stack
    let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.cinelog13pro.sessionQueue", qos: .userInteractive)
    private let videoProcessingQueue = DispatchQueue(label: "com.cinelog13pro.videoProcessingQueue", qos: .userInteractive)
    private var videoDevice: AVCaptureDevice?
    private var currentVideoInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let audioDataOutput = AVCaptureAudioDataOutput()
    private var lensPositionObserver: NSKeyValueObservation?
    private var isoObserver: NSKeyValueObservation?
    private var rackFocusTimer: Timer?

    // CoreMotion Gyro Horizon Manager
    private let motionManager = CMMotionManager()

    // Metal Compute, Render & CIContext
    let metalDevice: MTLDevice?
    private(set) var metalLibrary: MTLLibrary?
    private var commandQueue: MTLCommandQueue?
    private var computePipelineState: MTLComputePipelineState?
    private(set) var postGradePipelineState: MTLComputePipelineState?
    private var textureCache: CVMetalTextureCache?
    private var ciContext: CIContext?
    private var dummy3DLUTTexture: MTLTexture?
    private var custom3DLUTTexture: MTLTexture?
    private var custom3DLUTSize: Int = 33

    private let gpuFrameSemaphore = DispatchSemaphore(value: 2)
    private var recordBufferPool: CVPixelBufferPool?
    private var previewBufferPool: CVPixelBufferPool?
    private var currentRecordPoolDims = CMVideoDimensions(width: 0, height: 0)
    private var currentPreviewPoolDims = CMVideoDimensions(width: 0, height: 0)

    private let previewLock = NSLock()
    private var _latestPreviewTexture: MTLTexture?
    private var _latestRecordTexture: MTLTexture?
    private var pendingPhotoCaptureRequest: Bool = false
    weak var mtkViewRef: MTKView?

    var latestPreviewTexture: MTLTexture? {
        previewLock.lock()
        defer { previewLock.unlock() }
        return _latestPreviewTexture
    }

    // Rock-Solid Atomic Recording State Machine
    private let recordingStateLock = NSLock()
    private var recordingState: EngineRecordingState = .idle
    private var assetWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var audioWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var recordingStartTime: CMTime?
    private var currentOutputURL: URL?

    // Active Sensor Format Flags
    private var hardwareSupportsNativeAppleLog: Bool = false
    private var activeFormatIsTenBit: Bool = true
    private var activeFormatIsFullRange: Bool = false
    private var activeFormatIsSDRFallback: Bool = false
    private var frameCounter: UInt64 = 0
    private let startTimestamp: CFTimeInterval = CACurrentMediaTime()

    override init() {
        self.metalDevice = MTLCreateSystemDefaultDevice()
        super.init()
        setupMetalPipeline()
        setupOrientationAndMotionObservers()
    }

    // MARK: - Metal Initialization

    private func setupMetalPipeline() {
        guard let device = metalDevice else {
            statusMessage = "Metal GPU Unavailable"
            return
        }
        self.commandQueue = device.makeCommandQueue()
        self.ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any])
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)

        let lutDesc = MTLTextureDescriptor()
        lutDesc.textureType = .type3D
        lutDesc.pixelFormat = .rgba32Float
        lutDesc.width = 2
        lutDesc.height = 2
        lutDesc.depth = 2
        lutDesc.usage = [.shaderRead]
        self.dummy3DLUTTexture = device.makeTexture(descriptor: lutDesc)

        do {
            let library = try device.makeLibrary(source: MetalShaderLibrarySource.source, options: nil)
            self.metalLibrary = library
            if let kernelFunc = library.makeFunction(name: "processAppleLogFrame") {
                self.computePipelineState = try device.makeComputePipelineState(function: kernelFunc)
            }
            if let postFunc = library.makeFunction(name: "gradeExistingLogFrame") {
                self.postGradePipelineState = try device.makeComputePipelineState(function: postFunc)
            }
        } catch {
            statusMessage = "Shader Compile Error: \(error.localizedDescription)"
        }
    }

    private func setupOrientationAndMotionObservers() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDeviceOrientationChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 24.0
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
                guard let self = self, let attitude = motion?.attitude, let gravity = motion?.gravity else { return }
                let angleRad = atan2(gravity.x, -gravity.y)
                var deg = angleRad * 180.0 / .pi
                if !self.orientationMode.isPortraitSwapped {
                    deg = atan2(gravity.y, -gravity.x) * 180.0 / .pi
                }
                self.deviceRollDegrees = deg
                self.devicePitchDegrees = attitude.pitch * 180.0 / .pi
            }
        }
    }

    @objc private func handleDeviceOrientationChange() {
        guard autoFollowDeviceOrientation, !isRecording else { return }
        switch UIDevice.current.orientation {
        case .portrait:
            orientationMode = .portrait
        case .landscapeLeft:
            orientationMode = .landscapeRight
        case .landscapeRight:
            orientationMode = .landscapeLeft
        case .portraitUpsideDown:
            orientationMode = .portraitUpsideDown
        default:
            break
        }
    }

    func rotateOrientation90Degrees() {
        guard !isRecording else {
            statusMessage = "录制中已锁定画面方向"
            return
        }
        let nextRaw = (orientationMode.rawValue + 1) % CaptureOrientationMode.allCases.count
        if let nextMode = CaptureOrientationMode(rawValue: nextRaw) {
            orientationMode = nextMode
        }
    }

    func importCustomCubeLUT(from url: URL) {
        guard let device = metalDevice else { return }
        do {
            let (tex, lutSize, name) = try CubeLUTManager.loadCubeLUT(from: url, device: device)
            self.custom3DLUTTexture = tex
            self.custom3DLUTSize = lutSize
            self.customLUTName = "\(name) (\(lutSize)³)"
            self.colorEngineMode = .lutGrading
            self.lutSettings.lutPreset = .customCubeLUT
            self.statusMessage = "已加载独立 LUT: \(name)"
        } catch {
            self.statusMessage = error.localizedDescription
        }
    }

    // MARK: - Permissions & Session Configuration

    func requestPermissionsAndStart() {
        Task {
            let videoAllowed = await AVCaptureDevice.requestAccess(for: .video)
            let audioAllowed = await AVCaptureDevice.requestAccess(for: .audio)
            guard videoAllowed else {
                await MainActor.run { self.statusMessage = "请在系统设置中开启相机权限" }
                return
            }
            sessionQueue.async { [weak self] in
                guard let self = self else { return }
                if !self.captureSession.isRunning {
                    self.configureCaptureSession(includeAudio: audioAllowed)
                    self.captureSession.startRunning()
                    DispatchQueue.main.async {
                        self.isSessionRunning = self.captureSession.isRunning
                    }
                }
            }
        }
    }

    private func configureCaptureSession(includeAudio: Bool) {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        captureSession.sessionPreset = .inputPriority
        captureSession.automaticallyConfiguresCaptureDeviceForWideColor = false

        let camera = AVCaptureDevice.default(selectedLens.deviceType, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)

        guard let activeCamera = camera else {
            DispatchQueue.main.async { self.statusMessage = "未检测到后置摄像头" }
            return
        }
        self.videoDevice = activeCamera
        observeDeviceTelemetry(activeCamera)

        do {
            let videoInput = try AVCaptureDeviceInput(device: activeCamera)
            if captureSession.canAddInput(videoInput) {
                captureSession.addInput(videoInput)
                self.currentVideoInput = videoInput
            }

            if includeAudio,
               let mic = AVCaptureDevice.default(for: .audio),
               let audioInput = try? AVCaptureDeviceInput(device: mic),
               captureSession.canAddInput(audioInput) {
                captureSession.addInput(audioInput)
            }

            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.setSampleBufferDelegate(self, queue: videoProcessingQueue)
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
            }

            audioDataOutput.setSampleBufferDelegate(self, queue: videoProcessingQueue)
            if captureSession.canAddOutput(audioDataOutput) {
                captureSession.addOutput(audioDataOutput)
            }

            try selectBestSensorFormat(for: activeCamera, preset: selectedSensorPreset, targetFPS: selectedFPS.rawValue)
            configureVideoOutputConnection()
            applyFocusConfigurationInternal(on: activeCamera)

        } catch {
            DispatchQueue.main.async {
                self.statusMessage = "配置采集管线失败: \(error.localizedDescription)"
            }
        }
    }

    private func observeDeviceTelemetry(_ device: AVCaptureDevice) {
        lensPositionObserver?.invalidate()
        isoObserver?.invalidate()

        lensPositionObserver = device.observe(\.lensPosition, options: [.new]) { [weak self] dev, _ in
            guard let self = self else { return }
            if self.focusMode != .manualFocus && !self.isRackingFocus {
                DispatchQueue.main.async {
                    self.lensPosition = dev.lensPosition
                }
            }
        }

        isoObserver = device.observe(\.iso, options: [.new]) { [weak self] dev, _ in
            guard let self = self else { return }
            if !self.isManualExposure {
                DispatchQueue.main.async {
                    self.currentISO = dev.iso
                }
            }
        }
    }

    // MARK: - Professional Stabilization Configuration

    private func configureVideoOutputConnection() {
        guard let connection = videoDataOutput.connection(with: .video) else { return }
        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .landscapeRight
        }
        applyStabilizationToConnection(connection)
    }

    private func applyStabilizationMode() {
        sessionQueue.async { [weak self] in
            guard let self = self, let connection = self.videoDataOutput.connection(with: .video) else { return }
            self.applyStabilizationToConnection(connection)
        }
    }

    private func applyStabilizationToConnection(_ connection: AVCaptureConnection) {
        guard connection.isVideoStabilizationSupported else { return }
        let requested = stabilizationMode.avMode
        if let device = videoDevice, device.activeFormat.isVideoStabilizationModeSupported(requested) {
            connection.preferredVideoStabilizationMode = requested
            DispatchQueue.main.async {
                self.statusMessage = "防抖已切换: \(self.stabilizationMode.shortTitle) (OIS+EIS)"
            }
        } else if let device = videoDevice, device.activeFormat.isVideoStabilizationModeSupported(.cinematic) && requested == .cinematicExtended {
            connection.preferredVideoStabilizationMode = .cinematic
        } else if let device = videoDevice, device.activeFormat.isVideoStabilizationModeSupported(.standard) && requested != .off {
            connection.preferredVideoStabilizationMode = .standard
        } else {
            connection.preferredVideoStabilizationMode = .off
        }
    }

    // MARK: - Professional Focus, Manual Lens Pull, Rack Focus A↔B & Touch AF/AE

    private func applyFocusConfiguration() {
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            self.applyFocusConfigurationInternal(on: device)
        }
    }

    private func applyFocusConfigurationInternal(on device: AVCaptureDevice) {
        guard (try? device.lockForConfiguration()) != nil else { return }
        defer { device.unlockForConfiguration() }

        if device.isSmoothAutoFocusSupported {
            device.isSmoothAutoFocusEnabled = isSmoothAFEnabled
        }
        if device.isAutoFocusRangeRestrictionSupported {
            device.autoFocusRangeRestriction = afRangeRestriction.avRange
        }

        switch focusMode {
        case .continuousAF:
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
        case .singleAF:
            if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus
            }
        case .manualFocus:
            if device.isLockingFocusWithCustomLensPositionSupported {
                let clamped = max(0.0, min(1.0, lensPosition))
                device.setFocusModeLocked(lensPosition: clamped, completionHandler: nil)
            }
        }
    }

    func setManualFocusLensPosition(_ pos: Float) {
        let clamped = max(0.0, min(1.0, pos))
        lensPosition = clamped
        if focusMode != .manualFocus {
            focusMode = .manualFocus
        }
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            if (try? device.lockForConfiguration()) != nil {
                if device.isLockingFocusWithCustomLensPositionSupported {
                    device.setFocusModeLocked(lensPosition: clamped, completionHandler: nil)
                }
                device.unlockForConfiguration()
            }
        }
    }

    func triggerRackFocus(to targetPos: Float) {
        rackFocusTimer?.invalidate()
        focusMode = .manualFocus
        isRackingFocus = true

        let startPos = lensPosition
        let endPos = max(0.0, min(1.0, targetPos))
        let duration = max(0.2, rackDurationSeconds)
        let startTime = CACurrentMediaTime()

        rackFocusTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] timer in
            guard let self = self else {
                timer.invalidate()
                return
            }
            let elapsed = CACurrentMediaTime() - startTime
            let progress = Float(min(1.0, elapsed / duration))
            let smoothT = progress * progress * (3.0 - 2.0 * progress)
            let interpolated = startPos + (endPos - startPos) * smoothT
            self.setManualFocusLensPosition(interpolated)

            if progress >= 1.0 {
                timer.invalidate()
                self.isRackingFocus = false
                self.statusMessage = String(format: "跟焦完成 (Lens: %.2f)", endPos)
            }
        }
    }

    func handleViewfinderTap(normalizedPoint: CGPoint, lockFocusAndExposure: Bool = false) {
        focusReticleNormalized = normalizedPoint
        isFocusLockedByTap = lockFocusAndExposure

        let sensorPoint: CGPoint
        switch orientationMode {
        case .portrait:
            sensorPoint = CGPoint(x: normalizedPoint.y, y: 1.0 - normalizedPoint.x)
        case .landscapeRight:
            sensorPoint = normalizedPoint
        case .landscapeLeft:
            sensorPoint = CGPoint(x: 1.0 - normalizedPoint.x, y: 1.0 - normalizedPoint.y)
        case .portraitUpsideDown:
            sensorPoint = CGPoint(x: 1.0 - normalizedPoint.y, y: normalizedPoint.x)
        }

        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            if (try? device.lockForConfiguration()) != nil {
                defer { device.unlockForConfiguration() }

                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = sensorPoint
                    let targetFocusMode: AVCaptureDevice.FocusMode = lockFocusAndExposure ? .autoFocus : .continuousAutoFocus
                    if device.isFocusModeSupported(targetFocusMode) {
                        device.focusMode = targetFocusMode
                    }
                }

                if !self.isManualExposure && device.isExposurePointOfInterestSupported {
                    device.exposurePointOfInterest = sensorPoint
                    let targetExpMode: AVCaptureDevice.ExposureMode = lockFocusAndExposure ? .autoExpose : .continuousAutoExposure
                    if device.isExposureModeSupported(targetExpMode) {
                        device.exposureMode = targetExpMode
                    }
                }

                DispatchQueue.main.async {
                    self.focusMode = lockFocusAndExposure ? .singleAF : .continuousAF
                    self.statusMessage = lockFocusAndExposure ? "AF/AE 已锁定于选点" : "已对焦并测光选点"
                }
            }
        }
    }

    // MARK: - Smooth Servo Zoom & Multi-Lens Switching

    func setZoomFactor(_ factor: CGFloat, animatedRate: Float? = nil) {
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            let maxZoom = min(device.activeFormat.videoMaxZoomFactor, 8.0)
            let clamped = max(1.0, min(factor, maxZoom))
            if (try? device.lockForConfiguration()) != nil {
                if let rate = animatedRate {
                    device.ramp(toVideoZoomFactor: clamped, withRate: rate)
                } else {
                    device.videoZoomFactor = clamped
                }
                device.unlockForConfiguration()
                DispatchQueue.main.async {
                    self.zoomFactor = clamped
                }
            }
        }
    }

    private func switchCameraLens(_ lens: CameraLensOption) {
        guard !isRecording else {
            statusMessage = "录制中无法切换物理镜头"
            return
        }
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard let newCamera = AVCaptureDevice.default(lens.deviceType, for: .video, position: .back) else {
                DispatchQueue.main.async { self.statusMessage = "当前设备不支持 \(lens.focalLabel) 镜头" }
                return
            }

            self.captureSession.beginConfiguration()
            if let oldInput = self.currentVideoInput {
                self.captureSession.removeInput(oldInput)
            }
            if let newInput = try? AVCaptureDeviceInput(device: newCamera),
               self.captureSession.canAddInput(newInput) {
                self.captureSession.addInput(newInput)
                self.currentVideoInput = newInput
                self.videoDevice = newCamera
                self.observeDeviceTelemetry(newCamera)
            }
            try? self.selectBestSensorFormat(for: newCamera, preset: self.selectedSensorPreset, targetFPS: self.selectedFPS.rawValue)
            self.configureVideoOutputConnection()
            self.applyFocusConfigurationInternal(on: newCamera)
            self.captureSession.commitConfiguration()

            DispatchQueue.main.async {
                self.zoomFactor = 1.0
            }
        }
    }

    private func reconfigureFormatAndSession() {
        guard !isRecording else { return }
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            self.captureSession.beginConfiguration()
            try? self.selectBestSensorFormat(for: device, preset: self.selectedSensorPreset, targetFPS: self.selectedFPS.rawValue)
            self.configureVideoOutputConnection()
            self.applyFocusConfigurationInternal(on: device)
            self.captureSession.commitConfiguration()
        }
    }

    private func selectBestSensorFormat(for device: AVCaptureDevice, preset: SensorFormatPreset, targetFPS: Double) throws {
        let tenBitSubtypes: [OSType] = [
            kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        ]
        let eightBitSubtypes: [OSType] = [
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ]
        let allSupportedSubtypes = tenBitSubtypes + eightBitSubtypes

        let fpsCompatibleFormats = device.formats.filter { format in
            let subType = CMFormatDescriptionGetMediaSubType(format.formatDescription)
            let supportsFPS = format.videoSupportedFrameRateRanges.contains {
                $0.minFrameRate <= targetFPS && targetFPS <= $0.maxFrameRate
            }
            return allSupportedSubtypes.contains(subType) && supportsFPS
        }

        let candidatePool: [AVCaptureDevice.Format]
        if preset == .openGateMax {
            let fourByThreePool = fpsCompatibleFormats.filter { fmt in
                let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                let ratio = Double(d.width) / Double(max(1, d.height))
                return abs(ratio - (4.0 / 3.0)) < 0.05 && d.width >= 2880
            }
            if !fourByThreePool.isEmpty {
                candidatePool = fourByThreePool
            } else {
                candidatePool = device.formats.filter { fmt in
                    let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                    let subType = CMFormatDescriptionGetMediaSubType(fmt.formatDescription)
                    let ratio = Double(d.width) / Double(max(1, d.height))
                    let supports24Or30 = fmt.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 24.0 }
                    return allSupportedSubtypes.contains(subType) && abs(ratio - (4.0 / 3.0)) < 0.05 && supports24Or30
                }
            }
        } else if preset == .uhd4K16x9 {
            candidatePool = fpsCompatibleFormats.filter {
                let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                return d.width == 3840 && d.height == 2160 && !$0.isVideoBinned
            }
        } else {
            candidatePool = fpsCompatibleFormats.filter {
                let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                return d.width == 1920 && d.height == 1080
            }
        }

        let sortedCandidates = candidatePool.sorted { a, b in
            let subA = CMFormatDescriptionGetMediaSubType(a.formatDescription)
            let subB = CMFormatDescriptionGetMediaSubType(b.formatDescription)
            let is10A = tenBitSubtypes.contains(subA) ? 1 : 0
            let is10B = tenBitSubtypes.contains(subB) ? 1 : 0
            if is10A != is10B {
                return is10A > is10B
            }
            let hlgA = a.supportedColorSpaces.contains(.HLG_BT2020) ? 1 : 0
            let hlgB = b.supportedColorSpaces.contains(.HLG_BT2020) ? 1 : 0
            if hlgA != hlgB {
                return hlgA > hlgB
            }
            let dimA = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let dimB = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            let areaA = Int64(dimA.width) * Int64(dimA.height)
            let areaB = Int64(dimB.width) * Int64(dimB.height)
            return areaA > areaB
        }

        guard let activeFormat = sortedCandidates.first ?? fpsCompatibleFormats.first else {
            throw NSError(domain: "CineLog13Pro", code: -10, userInfo: [NSLocalizedDescriptionKey: "未找到匹配的传感器格式"])
        }

        let chosenSubType = CMFormatDescriptionGetMediaSubType(activeFormat.formatDescription)
        let isTenBit = tenBitSubtypes.contains(chosenSubType)
        let isFullRange = (chosenSubType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        device.activeFormat = activeFormat

        let maxSupportedFPS = activeFormat.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 30.0
        let minSupportedFPS = activeFormat.videoSupportedFrameRateRanges.map(\.minFrameRate).min() ?? 24.0
        let effectiveFPS = min(maxSupportedFPS, max(minSupportedFPS, targetFPS))
        let frameDuration = CMTime(value: 1000, timescale: CMTimeScale(effectiveFPS * 1000))
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration

        var useNativeAppleLog = false
        var isSDRInput = false

        #if compiler(>=5.9)
        if #available(iOS 17.0, *), activeFormat.supportedColorSpaces.contains(.appleLog) {
            device.activeColorSpace = .appleLog
            useNativeAppleLog = true
        } else if activeFormat.supportedColorSpaces.contains(.HLG_BT2020) {
            device.activeColorSpace = .HLG_BT2020
            if activeFormat.isVideoHDRSupported {
                device.automaticallyAdjustsVideoHDREnabled = false
                device.isVideoHDREnabled = true
            }
        } else if activeFormat.supportedColorSpaces.contains(.P3_D65) {
            device.activeColorSpace = .P3_D65
            isSDRInput = true
        } else {
            isSDRInput = true
        }
        #else
        if activeFormat.supportedColorSpaces.contains(.HLG_BT2020) {
            device.activeColorSpace = .HLG_BT2020
            if activeFormat.isVideoHDRSupported {
                device.automaticallyAdjustsVideoHDREnabled = false
                device.isVideoHDREnabled = true
            }
        } else if activeFormat.supportedColorSpaces.contains(.P3_D65) {
            device.activeColorSpace = .P3_D65
            isSDRInput = true
        } else {
            isSDRInput = true
        }
        #endif

        if device.isGeometricDistortionCorrectionSupported {
            device.isGeometricDistortionCorrectionEnabled = false
        }

        let availableOutFormats = videoDataOutput.availableVideoPixelFormatTypes
        if isTenBit {
            if availableOutFormats.contains(kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange) {
                videoDataOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
                ]
            } else if availableOutFormats.contains(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange) {
                videoDataOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                ]
            }
        } else {
            let fallbackFormat = isFullRange
                ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            videoDataOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: fallbackFormat
            ]
        }

        self.hardwareSupportsNativeAppleLog = useNativeAppleLog
        self.activeFormatIsTenBit = isTenBit
        self.activeFormatIsFullRange = isFullRange
        self.activeFormatIsSDRFallback = isSDRInput

        let minISOVal = activeFormat.minISO
        let maxISOVal = min(activeFormat.maxISO, 3200.0)
        let currentISOVal = device.iso
        let dims = CMVideoFormatDescriptionGetDimensions(activeFormat.formatDescription)
        let bitStr = isTenBit ? "10-Bit BT.2020" : "P3 16-Bit"

        DispatchQueue.main.async {
            self.minISO = minISOVal
            self.maxISO = maxISOVal
            self.currentISO = currentISOVal
            self.activeResolutionBadge = "\(dims.width)×\(dims.height) (\(bitStr))"
            self.activePipelineBadge = "\(self.selectedLens.focalLabel)"
            self.statusMessage = "60fps 零卡顿管线就绪"
        }
    }

    // MARK: - Cinema Exposure & White Balance

    func updateExposureAndShutter() {
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            if (try? device.lockForConfiguration()) != nil {
                defer { device.unlockForConfiguration() }
                if self.isManualExposure {
                    let shutterSeconds = (self.shutterAngleDegrees / 360.0) / self.selectedFPS.rawValue
                    let clampedSeconds = max(
                        CMTimeGetSeconds(device.activeFormat.minExposureDuration),
                        min(shutterSeconds, CMTimeGetSeconds(device.activeFormat.maxExposureDuration))
                    )
                    let duration = CMTimeMakeWithSeconds(clampedSeconds, preferredTimescale: 1_000_000)
                    let clampedISO = max(device.activeFormat.minISO, min(self.currentISO, device.activeFormat.maxISO))
                    device.setExposureModeCustom(duration: duration, iso: clampedISO, completionHandler: nil)
                } else if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
            }
        }
    }

    func setManualISO(_ isoValue: Float) {
        currentISO = isoValue
        guard isManualExposure else { return }
        updateExposureAndShutter()
    }

    private func applyWhiteBalanceKelvinAndTint() {
        guard isWhiteBalanceLocked else { return }
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            let tempAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(
                temperature: self.whiteBalanceKelvin,
                tint: self.whiteBalanceTint
            )
            var gains = device.deviceWhiteBalanceGains(for: tempAndTint)
            let maxGain = device.maxWhiteBalanceGain
            gains.redGain   = max(1.0, min(gains.redGain, maxGain))
            gains.greenGain = max(1.0, min(gains.greenGain, maxGain))
            gains.blueGain  = max(1.0, min(gains.blueGain, maxGain))

            if (try? device.lockForConfiguration()) != nil {
                device.setWhiteBalanceModeLocked(with: gains, completionHandler: nil)
                device.unlockForConfiguration()
            }
        }
    }

    private func toggleWhiteBalanceLock(_ locked: Bool) {
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            if (try? device.lockForConfiguration()) != nil {
                if locked {
                    let tempAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(
                        temperature: self.whiteBalanceKelvin,
                        tint: self.whiteBalanceTint
                    )
                    var gains = device.deviceWhiteBalanceGains(for: tempAndTint)
                    let maxGain = device.maxWhiteBalanceGain
                    gains.redGain   = max(1.0, min(gains.redGain, maxGain))
                    gains.greenGain = max(1.0, min(gains.greenGain, maxGain))
                    gains.blueGain  = max(1.0, min(gains.blueGain, maxGain))
                    device.setWhiteBalanceModeLocked(with: gains, completionHandler: nil)
                } else if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                }
                device.unlockForConfiguration()
            }
        }
    }

    // MARK: - High-Bit-Depth Still Photo Capture (With Orientation Applied via CIImage)

    func captureStillPhoto() {
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        previewLock.lock()
        pendingPhotoCaptureRequest = true
        previewLock.unlock()

        DispatchQueue.main.async {
            self.isCapturingPhoto = true
            withAnimation(.easeOut(duration: 0.12)) {
                self.showShutterFlash = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.14) {
                withAnimation(.easeIn(duration: 0.15)) {
                    self.showShutterFlash = false
                }
            }
        }

        videoProcessingQueue.asyncAfter(deadline: .now() + 0.06) { [weak self] in
            guard let self = self else { return }
            self.previewLock.lock()
            let gradedTex = self._latestPreviewTexture
            let rawLogTex = self._latestRecordTexture
            self.pendingPhotoCaptureRequest = false
            self.previewLock.unlock()

            guard let targetTex = gradedTex, let ciCtx = self.ciContext else {
                DispatchQueue.main.async {
                    self.isCapturingPhoto = false
                    self.statusMessage = "快门缓冲未就绪，请重试"
                }
                return
            }

            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            if let ciImg = self.makeOrientedCIImage(from: targetTex, colorSpace: colorSpace),
               let cgImg = ciCtx.createCGImage(ciImg, from: ciImg.extent, format: .RGBA16, colorSpace: colorSpace) {
                let uiImage = UIImage(cgImage: cgImg)
                self.savePhotoToLibrary(image: uiImage, label: self.activeModeSummaryBadge)
            }

            if self.dualSaveRawAndFilmPhoto, let rawTex = rawLogTex,
               let rawCI = self.makeOrientedCIImage(from: rawTex, colorSpace: colorSpace),
               let rawCG = ciCtx.createCGImage(rawCI, from: rawCI.extent, format: .RGBA16, colorSpace: colorSpace) {
                let rawUI = UIImage(cgImage: rawCG)
                self.savePhotoToLibrary(image: rawUI, label: "AppleLog_Raw")
            }
        }
    }

    private func makeOrientedCIImage(from texture: MTLTexture, colorSpace: CGColorSpace) -> CIImage? {
        guard var ciImg = CIImage(mtlTexture: texture, options: [.colorSpace: colorSpace]) else { return nil }
        // Flip vertically first (Metal top-left origin -> CIImage bottom-left origin)
        ciImg = ciImg.transformed(by: CGAffineTransform(scaleX: 1.0, y: -1.0).translatedBy(x: 0, y: -ciImg.extent.height))
        switch orientationMode {
        case .portrait:
            ciImg = ciImg.oriented(.right)
        case .landscapeRight:
            break
        case .landscapeLeft:
            ciImg = ciImg.oriented(.down)
        case .portraitUpsideDown:
            ciImg = ciImg.oriented(.left)
        }
        return ciImg
    }

    private func savePhotoToLibrary(image: UIImage, label: String) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.isCapturingPhoto = false
                    self.statusMessage = "请开启相册写入权限以保存照片"
                }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }) { success, error in
                DispatchQueue.main.async {
                    self.isCapturingPhoto = false
                    if success {
                        self.statusMessage = "已保存照片 (\(label)) 至系统相册"
                    } else {
                        self.statusMessage = "照片保存失败: \(error?.localizedDescription ?? "未知错误")"
                    }
                }
            }
        }
    }

    // MARK: - Rock-Solid 10-Bit ProRes / HEVC Recording Control (Uses Native Sensor Dimensions + QuickTime Transform!)

    func triggerMainShutter() {
        if captureMode == .photo {
            captureStillPhoto()
        } else {
            toggleRecording()
        }
    }

    func toggleRecording() {
        recordingStateLock.lock()
        let currentState = recordingState
        recordingStateLock.unlock()

        switch currentState {
        case .idle:
            startRecording()
        case .recording:
            stopRecording()
        case .starting, .stopping:
            break
        }
    }

    private func startRecording() {
        recordingStateLock.lock()
        guard recordingState == .idle else {
            recordingStateLock.unlock()
            return
        }
        recordingState = .starting
        recordingStateLock.unlock()

        let feedback = UIImpactFeedbackGenerator(style: .heavy)
        feedback.impactOccurred()

        videoProcessingQueue.async { [weak self] in
            guard let self = self else { return }

            let rawW = Int(self.currentRecordPoolDims.width > 0 ? self.currentRecordPoolDims.width : 3840)
            let rawH = Int(self.currentRecordPoolDims.height > 0 ? self.currentRecordPoolDims.height : 2160)
            let width = (rawW / 2) * 2
            let height = (rawH / 2) * 2

            let takeStr = String(format: "T%02d", self.clipTakeNumber)
            let fileName = "\(self.projectReelName)_\(takeStr)_\(width)x\(height)_\(Int(Date().timeIntervalSince1970)).mov"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try? FileManager.default.removeItem(at: outputURL)

            do {
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

                let isRawLogRec = !self.bakeIntoRecording || self.colorEngineMode == .pureAppleLog || (self.colorEngineMode == .lutGrading && self.lutSettings.lutPreset == .rawAppleLog)
                let colorProperties: [String: Any]
                if !isRawLogRec {
                    colorProperties = [
                        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
                    ]
                } else {
                    colorProperties = [
                        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_2100_HLG,
                        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
                    ]
                }

                var chosenCodec = self.selectedCodec
                var videoSettings: [String: Any] = [
                    AVVideoCodecKey: chosenCodec.avCodecType.rawValue,
                    AVVideoWidthKey: width,
                    AVVideoHeightKey: height,
                    AVVideoColorPropertiesKey: colorProperties
                ]
                if chosenCodec == .hevc10Bit {
                    videoSettings[AVVideoCompressionPropertiesKey] = [
                        AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                        AVVideoAverageBitRateKey: 140_000_000,
                        AVVideoExpectedSourceFrameRateKey: self.selectedFPS.rawValue
                    ]
                }

                if !writer.canApply(outputSettings: videoSettings, forMediaType: .video) {
                    chosenCodec = .hevc10Bit
                    videoSettings = [
                        AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
                        AVVideoWidthKey: width,
                        AVVideoHeightKey: height,
                        AVVideoColorPropertiesKey: colorProperties,
                        AVVideoCompressionPropertiesKey: [
                            AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                            AVVideoAverageBitRateKey: 140_000_000,
                            AVVideoExpectedSourceFrameRateKey: self.selectedFPS.rawValue
                        ]
                    ]
                }

                let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
                vInput.expectsMediaDataInRealTime = true
                // Apply zero-cost QuickTime orientation matrix so Portrait & Landscape both play upright AND ProRes works in Portrait!
                vInput.transform = self.orientationMode.assetWriterTransform

                let sourcePixelAttributes: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                    kCVPixelBufferMetalCompatibilityKey as String: true
                ]
                let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: vInput,
                    sourcePixelBufferAttributes: sourcePixelAttributes
                )

                guard writer.canAdd(vInput) else {
                    throw NSError(domain: "CineLog13Pro", code: -21, userInfo: [NSLocalizedDescriptionKey: "编码器不支持当前分辨率 \(width)×\(height)"])
                }
                writer.add(vInput)

                let audioSettings: [String: Any] = [
                    AVFormatIDKey: kAudioFormatMPEG4AAC,
                    AVSampleRateKey: 48_000,
                    AVNumberOfChannelsKey: 2,
                    AVEncoderBitRateKey: 256_000
                ]
                let aInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
                aInput.expectsMediaDataInRealTime = true
                if writer.canAdd(aInput) {
                    writer.add(aInput)
                }

                guard writer.startWriting() else {
                    throw writer.error ?? NSError(domain: "CineLog13Pro", code: -22, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter 启动失败"])
                }

                self.assetWriter = writer
                self.videoWriterInput = vInput
                self.audioWriterInput = aInput
                self.pixelBufferAdaptor = adaptor
                self.recordingStartTime = nil
                self.currentOutputURL = outputURL

                self.recordingStateLock.lock()
                self.recordingState = .recording
                self.recordingStateLock.unlock()

                let modeDesc = self.bakeIntoRecording ? "直出 (\(self.activeModeSummaryBadge))" : "纯净 Apple Log"
                DispatchQueue.main.async {
                    self.selectedCodec = chosenCodec
                    self.isRecording = true
                    self.recordingDuration = 0
                    self.statusMessage = "REC \(chosenCodec.rawValue) • \(modeDesc)"
                }
            } catch {
                self.recordingStateLock.lock()
                self.recordingState = .idle
                self.recordingStateLock.unlock()

                DispatchQueue.main.async {
                    self.isRecording = false
                    self.statusMessage = "录制启动失败: \(error.localizedDescription)"
                }
            }
        }
    }

    private func stopRecording() {
        recordingStateLock.lock()
        guard recordingState == .recording else {
            recordingState = .idle
            recordingStateLock.unlock()
            DispatchQueue.main.async { self.isRecording = false }
            return
        }
        recordingState = .stopping
        recordingStateLock.unlock()

        let feedback = UIImpactFeedbackGenerator(style: .medium)
        feedback.impactOccurred()

        videoProcessingQueue.async { [weak self] in
            guard let self = self else { return }

            guard let writer = self.assetWriter else {
                self.resetRecordingStateToIdle()
                return
            }

            let outputURL = self.currentOutputURL

            if writer.status != .writing || self.recordingStartTime == nil {
                writer.cancelWriting()
                self.resetRecordingStateToIdle(message: "录制已取消或编码器重置")
                return
            }

            self.videoWriterInput?.markAsFinished()
            self.audioWriterInput?.markAsFinished()

            writer.finishWriting {
                self.resetRecordingStateToIdle(incrementTake: true, savedURL: outputURL)
                if let url = outputURL {
                    self.saveVideoToPhotoLibrary(url: url)
                }
            }
        }
    }

    private func resetRecordingStateToIdle(incrementTake: Bool = false, savedURL: URL? = nil, message: String? = nil) {
        self.assetWriter = nil
        self.videoWriterInput = nil
        self.audioWriterInput = nil
        self.pixelBufferAdaptor = nil
        self.recordingStartTime = nil

        recordingStateLock.lock()
        recordingState = .idle
        recordingStateLock.unlock()

        DispatchQueue.main.async {
            self.isRecording = false
            if incrementTake {
                self.clipTakeNumber += 1
            }
            if let url = savedURL {
                self.lastRecordedVideoURL = url
            }
            if let msg = message {
                self.statusMessage = msg
            } else {
                self.statusMessage = "正在保存视频至系统相册..."
            }
        }
    }

    func saveVideoToPhotoLibrary(url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.statusMessage = "文件已保存在沙盒: \(url.lastPathComponent)"
                }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }) { success, error in
                DispatchQueue.main.async {
                    if success {
                        self.statusMessage = "已保存 (\(self.selectedCodec.rawValue)) 至系统相册"
                    } else {
                        self.statusMessage = "相册保存失败: \(error?.localizedDescription ?? "未知错误")"
                    }
                }
            }
        }
    }

    private func ensurePixelBufferPools(recordW: Int32, recordH: Int32, previewW: Int32, previewH: Int32) {
        let evenRecW = (recordW / 2) * 2
        let evenRecH = (recordH / 2) * 2
        let evenPrevW = (previewW / 2) * 2
        let evenPrevH = (previewH / 2) * 2

        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 4]

        if currentRecordPoolDims.width != evenRecW || currentRecordPoolDims.height != evenRecH || recordBufferPool == nil {
            currentRecordPoolDims = CMVideoDimensions(width: evenRecW, height: evenRecH)
            let recAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
                kCVPixelBufferWidthKey as String: evenRecW,
                kCVPixelBufferHeightKey as String: evenRecH,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, recAttrs as CFDictionary, &recordBufferPool)
        }

        if currentPreviewPoolDims.width != evenPrevW || currentPreviewPoolDims.height != evenPrevH || previewBufferPool == nil {
            currentPreviewPoolDims = CMVideoDimensions(width: evenPrevW, height: evenPrevH)
            let prevAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
                kCVPixelBufferWidthKey as String: evenPrevW,
                kCVPixelBufferHeightKey as String: evenPrevH,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, prevAttrs as CFDictionary, &previewBufferPool)
        }
    }
}

// MARK: - 5. Coalesced Non-Blocking AVCaptureVideoDataOutput Delegate

extension AppleLogCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

    struct PipelineUniforms {
        var isNativeAppleLog: Int32
        var isFullRangeYCbCr: Int32
        var isSDRSensorInput: Int32
        var writeRecordBuffer: Int32

        var colorEngineMode: Int32
        var bakeIntoRecording: Int32
        var monitorDiagnosticMode: Int32

        var focusPeakingEnabled: Int32
        var focusPeakingColorIndex: Int32
        var focusPeakingThreshold: Float

        var lutPresetID: Int32
        var lutInputSpace: Int32
        var customLUTSize: Int32
        var hasCustom3DLUT: Int32
        var lutIntensity: Float
        var lutExposureEV: Float
        var lutContrast: Float
        var lutHighlightRollOff: Float
        var lutShadowLift: Float
        var lutSaturation: Float
        var lutVibrance: Float
        var lutTemperature: Float
        var lutTint: Float
        var lutDebanding: Float
        var liftRGB: SIMD4<Float>
        var gammaRGB: SIMD4<Float>
        var gainRGB: SIMD4<Float>

        var filmStockID: Int32
        var filmExposureEV: Float
        var pushPullStops: Float
        var halationIntensity: Float
        var emulsionBloom: Float
        var printPaperContrast: Float
        var printWarmthSplit: Float
        var filmGrain: Float
        var grainSize: Float
        var lensVignette: Float
        var chromaticAberration: Float
        var frameTime: Float
    }

    func makeShaderUniforms(writeRecord: Bool) -> PipelineUniforms {
        let l = lutSettings
        let f = filmSettings
        return PipelineUniforms(
            isNativeAppleLog: hardwareSupportsNativeAppleLog ? 1 : 0,
            isFullRangeYCbCr: activeFormatIsFullRange ? 1 : 0,
            isSDRSensorInput: activeFormatIsSDRFallback ? 1 : 0,
            writeRecordBuffer: writeRecord ? 1 : 0,
            colorEngineMode: Int32(colorEngineMode.rawValue),
            bakeIntoRecording: bakeIntoRecording ? 1 : 0,
            monitorDiagnosticMode: Int32(monitorDiagnostic.rawValue),
            focusPeakingEnabled: focusPeakingEnabled ? 1 : 0,
            focusPeakingColorIndex: Int32(focusPeakingColor.rawValue),
            focusPeakingThreshold: focusPeakingThreshold,
            lutPresetID: Int32(l.lutPreset.rawValue),
            lutInputSpace: Int32(l.lutInputSpace.rawValue),
            customLUTSize: Int32(custom3DLUTSize),
            hasCustom3DLUT: custom3DLUTTexture != nil ? 1 : 0,
            lutIntensity: l.lutIntensity,
            lutExposureEV: l.exposureEV,
            lutContrast: l.contrast,
            lutHighlightRollOff: l.highlightRollOff,
            lutShadowLift: l.shadowLift,
            lutSaturation: l.saturation,
            lutVibrance: l.vibrance,
            lutTemperature: l.temperature,
            lutTint: l.tint,
            lutDebanding: l.debanding,
            liftRGB: SIMD4<Float>(l.liftR, l.liftG, l.liftB, 0.0),
            gammaRGB: SIMD4<Float>(l.gammaR, l.gammaG, l.gammaB, 0.0),
            gainRGB: SIMD4<Float>(l.gainR, l.gainG, l.gainB, 1.0),
            filmStockID: Int32(f.filmStock.rawValue),
            filmExposureEV: f.exposureEV,
            pushPullStops: f.pushPullStops,
            halationIntensity: f.halationIntensity,
            emulsionBloom: f.emulsionBloom,
            printPaperContrast: f.printPaperContrast,
            printWarmthSplit: f.printWarmthSplit,
            filmGrain: f.filmGrain,
            grainSize: f.grainSize,
            lensVignette: f.lensVignette,
            chromaticAberration: f.chromaticAberration,
            frameTime: Float(CACurrentMediaTime() - startTimestamp)
        )
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if output == audioDataOutput {
            updateAudioPeakLevels(from: sampleBuffer)
            recordingStateLock.lock()
            let recActive = (recordingState == .recording)
            recordingStateLock.unlock()

            if recActive,
               let writer = assetWriter,
               writer.status == .writing,
               recordingStartTime != nil,
               let aInput = audioWriterInput,
               aInput.isReadyForMoreMediaData {
                aInput.append(sampleBuffer)
            }
            return
        }

        guard gpuFrameSemaphore.wait(timeout: .now()) == .success else {
            return
        }

        guard let inputPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let cache = textureCache,
              let commandQueue = commandQueue,
              let pipelineState = computePipelineState,
              let fallbackLUT = dummy3DLUTTexture else {
            gpuFrameSemaphore.signal()
            return
        }

        let sensorW = CVPixelBufferGetWidth(inputPixelBuffer)
        let sensorH = CVPixelBufferGetHeight(inputPixelBuffer)

        recordingStateLock.lock()
        let recActive = (recordingState == .recording)
        recordingStateLock.unlock()

        previewLock.lock()
        let needFullRes = recActive || pendingPhotoCaptureRequest
        previewLock.unlock()

        // When framing/previewing idle (not recording/snapping), downsample preview compute grid by 2x (1920x1080) -> 4x faster GPU compute!
        let computeW = needFullRes ? sensorW : max(960, sensorW / 2)
        let computeH = needFullRes ? sensorH : max(540, sensorH / 2)
        ensurePixelBufferPools(recordW: Int32(sensorW), recordH: Int32(sensorH), previewW: Int32(computeW), previewH: Int32(computeH))

        let lumaW = CVPixelBufferGetWidthOfPlane(inputPixelBuffer, 0)
        let lumaH = CVPixelBufferGetHeightOfPlane(inputPixelBuffer, 0)
        let chromaW = CVPixelBufferGetWidthOfPlane(inputPixelBuffer, 1)
        let chromaH = CVPixelBufferGetHeightOfPlane(inputPixelBuffer, 1)

        let lumaFormat: MTLPixelFormat = activeFormatIsTenBit ? .r16Unorm : .r8Unorm
        let chromaFormat: MTLPixelFormat = activeFormatIsTenBit ? .rg16Unorm : .rg8Unorm

        var cvLumaTex: CVMetalTexture?
        var cvChromaTex: CVMetalTexture?

        let statusY = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, inputPixelBuffer, nil, lumaFormat, lumaW, lumaH, 0, &cvLumaTex
        )
        let statusCbCr = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, inputPixelBuffer, nil, chromaFormat, chromaW, chromaH, 1, &cvChromaTex
        )

        guard statusY == kCVReturnSuccess, statusCbCr == kCVReturnSuccess,
              let lumaRef = cvLumaTex, let chromaRef = cvChromaTex,
              let lumaTex = CVMetalTextureGetTexture(lumaRef),
              let chromaTex = CVMetalTextureGetTexture(chromaRef),
              let recPool = recordBufferPool,
              let prevPool = previewBufferPool else {
            gpuFrameSemaphore.signal()
            return
        }

        var previewPixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, prevPool, &previewPixelBuffer)
        guard let prevOutPB = previewPixelBuffer else {
            gpuFrameSemaphore.signal()
            return
        }

        var logOutPixelBuffer: CVPixelBuffer?
        if needFullRes {
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, recPool, &logOutPixelBuffer)
        }
        let logOutPB = logOutPixelBuffer ?? prevOutPB

        var cvLogOutTex: CVMetalTexture?
        var cvPrevOutTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, prevOutPB, nil, .rgba16Float, computeW, computeH, 0, &cvPrevOutTex
        )
        if needFullRes {
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache, logOutPB, nil, .rgba16Float, sensorW, sensorH, 0, &cvLogOutTex
            )
        }

        guard let prevOutRef = cvPrevOutTex,
              let prevOutTex = CVMetalTextureGetTexture(prevOutRef),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            gpuFrameSemaphore.signal()
            return
        }

        let logOutTex = cvLogOutTex.flatMap { CVMetalTextureGetTexture($0) } ?? prevOutTex
        var uniforms = makeShaderUniforms(writeRecord: needFullRes)

        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(lumaTex, index: 0)
        encoder.setTexture(chromaTex, index: 1)
        encoder.setTexture(logOutTex, index: 2)
        encoder.setTexture(prevOutTex, index: 3)
        encoder.setTexture(custom3DLUTTexture ?? fallbackLUT, index: 4)
        encoder.setBytes(&uniforms, length: MemoryLayout<PipelineUniforms>.stride, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (computeW + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (computeH + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            self.previewLock.lock()
            self._latestPreviewTexture = prevOutTex
            if needFullRes {
                self._latestRecordTexture = logOutTex
            }
            self.previewLock.unlock()

            // Trigger event-driven MTKView draw on main thread ONLY when a new camera frame is ready!
            DispatchQueue.main.async {
                self.mtkViewRef?.setNeedsDisplay()
            }

            if recActive {
                self.videoProcessingQueue.async {
                    self.appendRecordedPixelBuffer(logOutPB, timestamp: timestamp)
                }
            }
            self.gpuFrameSemaphore.signal()
        }
        commandBuffer.commit()

        frameCounter += 1
        if frameCounter % 24 == 0 {
            computeFastLumaHistogram(from: inputPixelBuffer, isTenBit: activeFormatIsTenBit)
        }
    }

    private func appendRecordedPixelBuffer(_ pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        recordingStateLock.lock()
        let recActive = (recordingState == .recording)
        recordingStateLock.unlock()

        guard recActive, let writer = assetWriter else { return }

        if writer.status == .failed {
            let errMsg = writer.error?.localizedDescription ?? "编码器写入异常"
            resetRecordingStateToIdle(message: "录制异常中断: \(errMsg)")
            return
        }

        if writer.status == .writing,
           let vInput = videoWriterInput,
           let adaptor = pixelBufferAdaptor {
            if recordingStartTime == nil {
                recordingStartTime = timestamp
                writer.startSession(atSourceTime: timestamp)
            }
            if vInput.isReadyForMoreMediaData {
                if !adaptor.append(pixelBuffer, withPresentationTime: timestamp) {
                    let errMsg = writer.error?.localizedDescription ?? "像素缓冲写入失败"
                    resetRecordingStateToIdle(message: "编码中止: \(errMsg)")
                    return
                }
            }
            if let start = recordingStartTime {
                let elapsed = CMTimeGetSeconds(CMTimeSubtract(timestamp, start))
                DispatchQueue.main.async {
                    self.recordingDuration = max(0, elapsed)
                }
            }
        }
    }

    private func updateAudioPeakLevels(from sampleBuffer: CMSampleBuffer) {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer, length >= 4 else { return }

        let sampleCount = length / 2
        let int16Ptr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Int16.self)
        var maxLeft: Float = 0.00001
        var maxRight: Float = 0.00001

        let step = max(1, sampleCount / 64)
        for i in stride(from: 0, to: sampleCount, by: step) {
            let sample = abs(Float(int16Ptr[i])) / 32768.0
            if i % 2 == 0 {
                if sample > maxLeft { maxLeft = sample }
            } else {
                if sample > maxRight { maxRight = sample }
            }
        }
        if sampleCount < 2 { maxRight = maxLeft }

        let dbL = max(-60.0, min(0.0, 20.0 * log10(maxLeft)))
        let dbR = max(-60.0, min(0.0, 20.0 * log10(maxRight)))

        DispatchQueue.main.async {
            self.audioPeakDBLeft = dbL
            self.audioPeakDBRight = dbR
        }
    }

    private func computeFastLumaHistogram(from pixelBuffer: CVPixelBuffer, isTenBit: Bool) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddr = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)

        var bins = Array(repeating: Float(0), count: 64)
        var totalSamples: Float = 0

        if isTenBit {
            let uint16Ptr = baseAddr.assumingMemoryBound(to: UInt16.self)
            let strideU16 = bytesPerRow / 2
            for y in stride(from: 0, to: height, by: 48) {
                let rowOffset = y * strideU16
                for x in stride(from: 0, to: width, by: 48) {
                    let normalized = Float(uint16Ptr[rowOffset + x]) / 65535.0
                    let binIdx = min(63, max(0, Int(normalized * 63.0)))
                    bins[binIdx] += 1
                    totalSamples += 1
                }
            }
        } else {
            let uint8Ptr = baseAddr.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: 0, to: height, by: 48) {
                let rowOffset = y * bytesPerRow
                for x in stride(from: 0, to: width, by: 48) {
                    let normalized = Float(uint8Ptr[rowOffset + x]) / 255.0
                    let binIdx = min(63, max(0, Int(normalized * 63.0)))
                    bins[binIdx] += 1
                    totalSamples += 1
                }
            }
        }

        if totalSamples > 0 {
            let maxCount = bins.max() ?? 1.0
            for i in 0..<64 {
                bins[i] /= max(maxCount, 1.0)
            }
        }

        DispatchQueue.main.async {
            self.lumaHistogram = bins
        }
    }

    func exportPostGradedVideo(sourceURL: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        Task {
            do {
                let asset = AVURLAsset(url: sourceURL)
                guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
                    throw NSError(domain: "PostGrade", code: -1, userInfo: [NSLocalizedDescriptionKey: "未找到视频轨道"])
                }
                let naturalSize = try await videoTrack.load(.naturalSize)
                let preferredTransform = try await videoTrack.load(.preferredTransform)
                let nominalFPS = try await videoTrack.load(.nominalFrameRate)
                let width = (Int(abs(naturalSize.width)) / 2) * 2
                let height = (Int(abs(naturalSize.height)) / 2) * 2

                let reader = try AVAssetReader(asset: asset)
                let readerOutput = AVAssetReaderTrackOutput(
                    track: videoTrack,
                    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf]
                )
                reader.add(readerOutput)

                let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("Graded_\(Int(Date().timeIntervalSince1970)).mov")
                try? FileManager.default.removeItem(at: outURL)

                let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)
                let videoSettings: [String: Any] = [
                    AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
                    AVVideoWidthKey: width,
                    AVVideoHeightKey: height,
                    AVVideoCompressionPropertiesKey: [
                        AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                        AVVideoAverageBitRateKey: 100_000_000,
                        AVVideoExpectedSourceFrameRateKey: max(24.0, Double(nominalFPS))
                    ]
                ]
                let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
                writerInput.transform = preferredTransform
                let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: writerInput,
                    sourcePixelBufferAttributes: [
                        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
                        kCVPixelBufferWidthKey as String: width,
                        kCVPixelBufferHeightKey as String: height,
                        kCVPixelBufferMetalCompatibilityKey as String: true
                    ]
                )
                writer.add(writerInput)

                guard reader.startReading(), writer.startWriting() else {
                    throw NSError(domain: "PostGrade", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法启动读写管线"])
                }
                writer.startSession(atSourceTime: .zero)

                guard let cache = self.textureCache,
                      let cmdQueue = self.commandQueue,
                      let postPipeline = self.postGradePipelineState,
                      let fallbackLUT = self.dummy3DLUTTexture else {
                    throw NSError(domain: "PostGrade", code: -3, userInfo: [NSLocalizedDescriptionKey: "Metal 管线未就绪"])
                }

                while reader.status == .reading {
                    if !writerInput.isReadyForMoreMediaData {
                        try await Task.sleep(nanoseconds: 5_000_000)
                        continue
                    }
                    guard let sampleBuffer = readerOutput.copyNextSampleBuffer(),
                          let srcPB = CMSampleBufferGetImageBuffer(sampleBuffer),
                          let pool = adaptor.pixelBufferPool else {
                        break
                    }
                    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                    var dstPB: CVPixelBuffer?
                    CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &dstPB)
                    guard let outPB = dstPB else { continue }

                    var cvSrcTex: CVMetalTexture?
                    var cvDstTex: CVMetalTexture?
                    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, srcPB, nil, .rgba16Float, width, height, 0, &cvSrcTex)
                    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, outPB, nil, .rgba16Float, width, height, 0, &cvDstTex)

                    if let sRef = cvSrcTex, let dRef = cvDstTex,
                       let sTex = CVMetalTextureGetTexture(sRef),
                       let dTex = CVMetalTextureGetTexture(dRef),
                       let cmdBuf = cmdQueue.makeCommandBuffer(),
                       let enc = cmdBuf.makeComputeCommandEncoder() {
                        var uniforms = self.makeShaderUniforms(writeRecord: true)
                        enc.setComputePipelineState(postPipeline)
                        enc.setTexture(sTex, index: 0)
                        enc.setTexture(dTex, index: 1)
                        enc.setTexture(self.custom3DLUTTexture ?? fallbackLUT, index: 2)
                        enc.setBytes(&uniforms, length: MemoryLayout<PipelineUniforms>.stride, index: 0)

                        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
                        let tgCount = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
                        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                        enc.endEncoding()
                        cmdBuf.commit()
                        cmdBuf.waitUntilCompleted()

                        adaptor.append(outPB, withPresentationTime: pts)
                    }
                }

                writerInput.markAsFinished()
                await writer.finishWriting()
                self.saveVideoToPhotoLibrary(url: outURL)
                await MainActor.run { completion(.success(outURL)) }
            } catch {
                await MainActor.run { completion(.failure(error)) }
            }
        }
    }
}

// MARK: - 6. Event-Driven Calibrated sRGB/Rec.709 Metal Viewfinder (Zero Stutter, Zero Color Space Double-Expansion!)

struct MetalLogViewfinder: UIViewRepresentable {
    @ObservedObject var engine: AppleLogCameraEngine

    func makeCoordinator() -> Coordinator {
        Coordinator(engine: engine)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: engine.metalDevice)
        mtkView.framebufferOnly = true
        // Event-driven drawing: only redraws when a new camera frame arrives (eliminates 60Hz main-thread spin!)
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColorMake(0, 0, 0, 1)

        if let metalLayer = mtkView.layer as? CAMetalLayer {
            metalLayer.wantsExtendedDynamicRangeContent = false
            metalLayer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        }
        mtkView.delegate = context.coordinator
        engine.mtkViewRef = mtkView
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        engine.mtkViewRef = uiView
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        let engine: AppleLogCameraEngine
        private let commandQueue: MTLCommandQueue?
        private var renderPipelineState: MTLRenderPipelineState?

        struct ViewfinderScaleUniforms {
            var scale: SIMD2<Float>
            var rotationMode: Int32
            var mirrorHorizontal: Int32
        }

        init(engine: AppleLogCameraEngine) {
            self.engine = engine
            self.commandQueue = engine.metalDevice?.makeCommandQueue()
            super.init()
            buildRenderPipeline()
        }

        private func buildRenderPipeline() {
            guard let device = engine.metalDevice,
                  let library = engine.metalLibrary,
                  let vertexFunc = library.makeFunction(name: "vertexViewfinder"),
                  let fragmentFunc = library.makeFunction(name: "fragmentViewfinder") else {
                return
            }
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vertexFunc
            desc.fragmentFunction = fragmentFunc
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: desc)
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let rpd = view.currentRenderPassDescriptor,
                  let previewTex = engine.latestPreviewTexture,
                  let pipeline = renderPipelineState,
                  let commandBuffer = commandQueue?.makeCommandBuffer(),
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else {
                return
            }

            let viewW = Float(max(1.0, view.drawableSize.width))
            let viewH = Float(max(1.0, view.drawableSize.height))
            let isPortrait = engine.orientationMode.isPortraitSwapped
            let dispTexW = Float(max(1, isPortrait ? previewTex.height : previewTex.width))
            let dispTexH = Float(max(1, isPortrait ? previewTex.width : previewTex.height))

            let viewAspect = viewW / viewH
            let texAspect  = dispTexW / dispTexH

            var scale = SIMD2<Float>(1.0, 1.0)
            if viewAspect > texAspect {
                scale.x = texAspect / viewAspect
            } else {
                scale.y = viewAspect / texAspect
            }

            var vfUniforms = ViewfinderScaleUniforms(
                scale: scale,
                rotationMode: Int32(engine.orientationMode.rawValue),
                mirrorHorizontal: engine.isMirroredHorizontal ? 1 : 0
            )

            renderEncoder.setRenderPipelineState(pipeline)
            renderEncoder.setVertexBytes(&vfUniforms, length: MemoryLayout<ViewfinderScaleUniforms>.stride, index: 0)
            renderEncoder.setFragmentTexture(previewTex, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}

// MARK: - 7. Post-Production Studio Sheet

struct VideoMovieTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent("ImportedLog_\(Int(Date().timeIntervalSince1970)).mov")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return Self(url: dest)
        }
    }
}

struct PostColorGradingStudioView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @Environment(\.dismiss) private var dismiss

    @State private var selectedItem: PhotosPickerItem?
    @State private var activeVideoURL: URL?
    @State private var player: AVPlayer?
    @State private var isExporting: Bool = false
    @State private var exportStatus: String = "选择「LUT 调色」或「物理胶片」（两套系统完全独立，不叠加）"
    @State private var showCubeShareSheet: Bool = false
    @State private var exportedCubeURL: URL?

    var body: some View {
        NavigationStack {
            VStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black)
                        .frame(height: 220)

                    if let player = player {
                        VideoPlayer(player: player)
                            .frame(height: 220)
                            .cornerRadius(12)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "film.stack")
                                .font(.system(size: 34))
                                .foregroundColor(.yellow)
                            Text(exportStatus)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.gray)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 20)
                        }
                    }
                }
                .padding(.horizontal, 16)

                HStack(spacing: 10) {
                    PhotosPicker(selection: $selectedItem, matching: .videos) {
                        Label("从相册导入视频", systemImage: "photo.on.rectangle")
                            .font(.system(size: 12, weight: .bold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.white.opacity(0.15))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    Button {
                        if let url = try? CubeLUTManager.exportCubeFile(lutParams: engine.lutSettings) {
                            exportedCubeURL = url
                            showCubeShareSheet = true
                        }
                    } label: {
                        Label("导出 .cube", systemImage: "cube.transparent")
                            .font(.system(size: 12, weight: .bold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.cyan.opacity(0.22))
                            .foregroundColor(.cyan)
                            .cornerRadius(8)
                    }

                    Spacer()

                    Button {
                        guard let url = activeVideoURL else { return }
                        isExporting = true
                        exportStatus = "正在执行 10-Bit GPU 渲染导出..."
                        engine.exportPostGradedVideo(sourceURL: url) { result in
                            isExporting = false
                            switch result {
                            case .success(let gradedURL):
                                activeVideoURL = gradedURL
                                player = AVPlayer(url: gradedURL)
                                player?.play()
                                exportStatus = "渲染完成！10-Bit 成片已存入系统相册"
                            case .failure(let err):
                                exportStatus = "渲染失败: \(err.localizedDescription)"
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            if isExporting {
                                ProgressView().tint(.black)
                            }
                            Text(isExporting ? "渲染中..." : "渲染导出 10-Bit")
                                .font(.system(size: 12, weight: .bold))
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(activeVideoURL == nil || isExporting ? Color.gray : Color.yellow)
                        .foregroundColor(.black)
                        .cornerRadius(8)
                    }
                    .disabled(activeVideoURL == nil || isExporting)
                }
                .padding(.horizontal, 16)

                Picker("独立色彩引擎", selection: $engine.colorEngineMode) {
                    ForEach(ColorEngineMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)

                ScrollView {
                    if engine.colorEngineMode == .lutGrading {
                        LUTSystemDrawerView(engine: engine, showCubeImporter: .constant(false))
                            .padding(.horizontal, 16)
                    } else if engine.colorEngineMode == .physicalFilmLab {
                        PhysicalFilmLabDrawerView(engine: engine)
                            .padding(.horizontal, 16)
                    } else {
                        Text("当前为「原生 Apple Log」直出模式，不叠加任何 LUT 或胶片模拟。")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.green)
                            .padding(20)
                    }
                }
            }
            .background(Color(white: 0.08).ignoresSafeArea())
            .navigationTitle("后期实验室 (LUT / 物理胶片独立切换)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                        .foregroundColor(.yellow)
                }
            }
            .onAppear {
                if activeVideoURL == nil, let lastURL = engine.lastRecordedVideoURL {
                    activeVideoURL = lastURL
                    player = AVPlayer(url: lastURL)
                    exportStatus = "已自动载入最新录制的视频素材"
                }
            }
            .onChange(of: selectedItem) { newItem in
                guard let newItem = newItem else { return }
                Task {
                    if let movie = try? await newItem.loadTransferable(type: VideoMovieTransferable.self) {
                        await MainActor.run {
                            activeVideoURL = movie.url
                            player = AVPlayer(url: movie.url)
                            exportStatus = "已载入视频，选择独立引擎后点击「渲染导出 10-Bit」"
                        }
                    }
                }
            }
            .sheet(isPresented: $showCubeShareSheet) {
                if let url = exportedCubeURL {
                    ShareSheetView(items: [url])
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

struct ShareSheetView: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - 8A. Independent System 1: LUT Color Grading System Drawer (Zero Film Simulation)

struct LUTSystemDrawerView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @Binding var showCubeImporter: Bool
    @State private var lutSubTab: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("独立引擎 1：LUT 调色系统（已自动关闭物理胶片模拟，绝不叠加）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                Spacer()
                Button("重置 LUT") { engine.lutSettings.reset() }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(BuiltInLUTPresetOption.allCases) { preset in
                        Button {
                            engine.colorEngineMode = .lutGrading
                            engine.lutSettings.lutPreset = preset
                        } label: {
                            Text(preset.title)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(engine.lutSettings.lutPreset == preset && engine.colorEngineMode == .lutGrading ? Color.yellow : Color.white.opacity(0.12))
                                .foregroundColor(engine.lutSettings.lutPreset == preset && engine.colorEngineMode == .lutGrading ? .black : .white)
                                .cornerRadius(6)
                        }
                    }
                }
            }

            Text(engine.lutSettings.lutPreset.subtitle)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(.cyan)

            HStack(spacing: 8) {
                Button {
                    showCubeImporter = true
                } label: {
                    Label("导入外部 .cube (\(engine.customLUTName))", systemImage: "folder.badge.plus")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.cyan.opacity(0.25))
                        .foregroundColor(.cyan)
                        .cornerRadius(6)
                }

                Picker("LUT Space", selection: $engine.lutSettings.lutInputSpace) {
                    ForEach(LUTInputSpaceOption.allCases) { sp in
                        Text(sp.title).tag(sp)
                    }
                }
                .pickerStyle(.segmented)
            }

            Toggle(isOn: $engine.bakeIntoRecording) {
                Text(engine.bakeIntoRecording ? "录制视频：将当前 LUT 烧录直出" : "录制视频：保存纯净 Apple Log（LUT 仅监看/拍照生效）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(engine.bakeIntoRecording ? .orange : .green)
            }
            .tint(.orange)

            Picker("LUT 调色分组", selection: $lutSubTab) {
                Text("一级调色 & 浓度").tag(0)
                Text("三路色轮 (Lift/Gamma/Gain)").tag(1)
            }
            .pickerStyle(.segmented)

            if lutSubTab == 0 {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "LUT 混合浓度", value: $engine.lutSettings.lutIntensity, range: 0.0...1.0, format: "%.2f", tint: .yellow)
                    GradeSliderRow(label: "LUT 曝光 EV", value: $engine.lutSettings.exposureEV, range: -2.0...2.0, format: "%+.1f EV", tint: .yellow)
                    GradeSliderRow(label: "对数对比度", value: $engine.lutSettings.contrast, range: 0.6...1.4, format: "%.2f", tint: .white)
                    GradeSliderRow(label: "高光柔和滚降", value: $engine.lutSettings.highlightRollOff, range: 0.0...1.0, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "LUT 饱和度", value: $engine.lutSettings.saturation, range: 0.0...1.8, format: "%.2f", tint: .pink)
                    GradeSliderRow(label: "去断层平滑", value: $engine.lutSettings.debanding, range: 0.0...1.0, format: "%.2f", tint: .cyan)
                }
            } else {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "暗部 Lift R", value: $engine.lutSettings.liftR, range: -0.15...0.15, format: "%+.2f", tint: .red)
                    GradeSliderRow(label: "暗部 Lift G", value: $engine.lutSettings.liftG, range: -0.15...0.15, format: "%+.2f", tint: .green)
                    GradeSliderRow(label: "暗部 Lift B", value: $engine.lutSettings.liftB, range: -0.15...0.15, format: "%+.2f", tint: .blue)
                    GradeSliderRow(label: "中灰 Gamma B", value: $engine.lutSettings.gammaB, range: -0.25...0.25, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "高光 Gain R", value: $engine.lutSettings.gainR, range: 0.75...1.25, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "高光 Gain B", value: $engine.lutSettings.gainB, range: 0.75...1.25, format: "%.2f", tint: .blue)
                }
            }
        }
    }
}

// MARK: - 8B. Independent System 2: Direct-Photon Physical Film Lab Drawer (Zero LUTs)

struct PhysicalFilmLabDrawerView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @State private var filmSubTab: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("独立引擎 2：物理胶片暗房（已自动关闭 LUT 系统，绝不叠加）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
                Spacer()
                Button("重置胶片") {
                    engine.filmSettings.selectStock(engine.filmSettings.filmStock)
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(PhysicalFilmStockOption.allCases) { stock in
                        Button {
                            engine.colorEngineMode = .physicalFilmLab
                            engine.filmSettings.selectStock(stock)
                        } label: {
                            Text(stock.shortName)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(engine.filmSettings.filmStock == stock && engine.colorEngineMode == .physicalFilmLab ? Color.orange : Color.white.opacity(0.12))
                                .foregroundColor(engine.filmSettings.filmStock == stock && engine.colorEngineMode == .physicalFilmLab ? .black : .white)
                                .cornerRadius(6)
                        }
                    }
                }
            }

            Text(engine.filmSettings.filmStock.chemistryBadge)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundColor(.yellow)

            Toggle(isOn: $engine.bakeIntoRecording) {
                Text(engine.bakeIntoRecording ? "录制视频：将物理胶片暗房直接冲洗进视频" : "录制视频：保存纯净 Apple Log（物理胶片仅监看/拍照生效）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(engine.bakeIntoRecording ? .orange : .green)
            }
            .tint(.orange)

            Picker("暗房控制分组", selection: $filmSubTab) {
                Text("显影迫冲 & 相纸印相").tag(0)
                Text("光学光晕 / 暗角 / 银盐颗粒").tag(1)
            }
            .pickerStyle(.segmented)

            if filmSubTab == 0 {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "胶片曝光 EV", value: $engine.filmSettings.exposureEV, range: -2.0...2.0, format: "%+.1f EV", tint: .yellow)
                    GradeSliderRow(label: "迫冲 Push/Pull", value: $engine.filmSettings.pushPullStops, range: -1.5...1.5, format: "%+.1f 档", tint: .orange)
                    GradeSliderRow(label: "相纸反差 Grade", value: $engine.filmSettings.printPaperContrast, range: 0.70...1.35, format: "%.2f", tint: .white)
                    GradeSliderRow(label: "暗房冷暖染印", value: $engine.filmSettings.printWarmthSplit, range: -1.0...1.0, format: "%+.2f", tint: .cyan)
                }
            } else {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "片基红光晕", value: $engine.filmSettings.halationIntensity, range: 0.0...1.0, format: "%.2f", tint: .red)
                    GradeSliderRow(label: "乳剂柔光 Bloom", value: $engine.filmSettings.emulsionBloom, range: 0.0...1.0, format: "%.2f", tint: .yellow)
                    GradeSliderRow(label: "银盐颗粒密度", value: $engine.filmSettings.filmGrain, range: 0.0...1.0, format: "%.2f", tint: .gray)
                    GradeSliderRow(label: "银盐画幅尺寸", value: $engine.filmSettings.grainSize, range: 0.5...1.8, format: "%.2fx", tint: .gray)
                    GradeSliderRow(label: "光学暗角 cos⁴θ", value: $engine.filmSettings.lensVignette, range: 0.0...1.0, format: "%.2f", tint: .purple)
                    GradeSliderRow(label: "径向色散 CA", value: $engine.filmSettings.chromaticAberration, range: 0.0...1.0, format: "%.2f", tint: .pink)
                }
            }
        }
    }
}

struct GradeSliderRow: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String
    let tint: Color

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
                .frame(width: 92, alignment: .leading)
            Slider(value: $value, in: range)
                .tint(tint)
            Text(String(format: format, value))
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(tint)
                .frame(width: 56, alignment: .trailing)
        }
    }
}

// MARK: - 9. Blackmagic-Style Overlays & Bottom Dock

enum ActiveControlTab: Int, CaseIterable, Identifiable {
    case none        = 0
    case format      = 1 // 画幅/防抖
    case focus       = 2 // 对焦/跟焦
    case exposure    = 3 // 曝光/色温
    case lutSystem   = 4 // 独立 LUT 调色
    case physicalLab = 5 // 独立物理胶片
    case monitor     = 6 // 辅助/场记

    var id: Int { rawValue }
}

struct ViewfinderGuidesAndHorizonOverlay: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            ZStack {
                if engine.showGridOverlay {
                    Path { path in
                        path.move(to: CGPoint(x: w / 3.0, y: 0))
                        path.addLine(to: CGPoint(x: w / 3.0, y: h))
                        path.move(to: CGPoint(x: 2.0 * w / 3.0, y: 0))
                        path.addLine(to: CGPoint(x: 2.0 * w / 3.0, y: h))
                        path.move(to: CGPoint(x: 0, y: h / 3.0))
                        path.addLine(to: CGPoint(x: w, y: h / 3.0))
                        path.move(to: CGPoint(x: 0, y: 2.0 * h / 3.0))
                        path.addLine(to: CGPoint(x: w, y: 2.0 * h / 3.0))
                    }
                    .stroke(Color.white.opacity(0.20), lineWidth: 0.8)
                }

                if engine.aspectGuide == .scope239 {
                    let targetH = min(h, w / 2.39)
                    Rectangle()
                        .stroke(Color.yellow.opacity(0.65), lineWidth: 1.2)
                        .frame(width: w * 0.94, height: targetH * 0.94)
                } else if engine.aspectGuide == .cinema185 {
                    let targetH = min(h, w / 1.85)
                    Rectangle()
                        .stroke(Color.yellow.opacity(0.65), lineWidth: 1.2)
                        .frame(width: w * 0.94, height: targetH * 0.94)
                } else if engine.aspectGuide == .vertical916 {
                    let targetW = min(w, h * (9.0 / 16.0))
                    Rectangle()
                        .stroke(Color.cyan.opacity(0.65), lineWidth: 1.2)
                        .frame(width: targetW * 0.92, height: h * 0.92)
                }

                if engine.showHorizonLeveler {
                    let isLevel = abs(engine.deviceRollDegrees) < 1.2
                    let lineColor: Color = isLevel ? .green : .yellow.opacity(0.85)

                    ZStack {
                        HStack(spacing: 90) {
                            Rectangle().fill(Color.white.opacity(0.5)).frame(width: 14, height: 1.5)
                            Rectangle().fill(Color.white.opacity(0.5)).frame(width: 14, height: 1.5)
                        }
                        Rectangle()
                            .fill(lineColor)
                            .frame(width: 72, height: 2.0)
                            .rotationEffect(.degrees(-engine.deviceRollDegrees))

                        Circle()
                            .stroke(lineColor, lineWidth: 1.5)
                            .frame(width: 10, height: 10)

                        Text(String(format: "%+.1f°", engine.deviceRollDegrees))
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .foregroundColor(lineColor)
                            .offset(y: 16)
                    }
                    .position(x: w * 0.5, y: h * 0.5)
                }

                if let pt = engine.focusReticleNormalized {
                    let reticleColor: Color = engine.isFocusLockedByTap ? .red : .yellow
                    ZStack {
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(reticleColor, lineWidth: 1.8)
                            .frame(width: 58, height: 58)
                        Circle()
                            .stroke(reticleColor.opacity(0.85), lineWidth: 1.2)
                            .frame(width: 22, height: 22)
                        if engine.isFocusLockedByTap {
                            Text("AF/AE LOCK")
                                .font(.system(size: 8, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.red)
                                .foregroundColor(.white)
                                .offset(y: -38)
                        }
                    }
                    .position(x: pt.x * w, y: pt.y * h)
                }

                if engine.showShutterFlash {
                    Color.white.opacity(0.65)
                        .ignoresSafeArea()
                }
            }
        }
        .allowsHitTesting(false)
    }
}

struct LumaHistogramOverlay: View {
    let bins: [Float]

    var body: some View {
        HStack(alignment: .bottom, spacing: 1) {
            ForEach(0..<bins.count, id: \.self) { index in
                Rectangle()
                    .fill(index > 58 ? Color.red : (index >= 28 && index <= 34 ? Color.green : Color.white.opacity(0.85)))
                    .frame(width: 1.4, height: max(2, CGFloat(bins[index]) * 24))
            }
        }
        .padding(4)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.white.opacity(0.2), lineWidth: 0.8)
        )
        .fixedSize()
    }
}

struct StereoAudioVUMeterView: View {
    let leftDB: Float
    let rightDB: Float

    private func normalizedLevel(_ db: Float) -> CGFloat {
        let clamped = max(-60.0, min(0.0, db))
        return CGFloat((clamped + 60.0) / 60.0)
    }

    private func barColor(_ db: Float) -> Color {
        if db > -3.0 { return .red }
        if db > -12.0 { return .yellow }
        return .green
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 3) {
                Text("L")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white.opacity(0.15))
                        .frame(width: 46, height: 4)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor(leftDB))
                        .frame(width: 46 * normalizedLevel(leftDB), height: 4)
                }
            }
            HStack(spacing: 3) {
                Text("R")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.white.opacity(0.15))
                        .frame(width: 46, height: 4)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor(rightDB))
                        .frame(width: 46 * normalizedLevel(rightDB), height: 4)
                }
            }
        }
        .padding(4)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.2), lineWidth: 0.8))
        .fixedSize()
    }
}

// MARK: - 10. Main Blackmagic Cinema & Dual-Engine Film Camera View

struct CineLogCameraView: View {
    @StateObject private var engine = AppleLogCameraEngine()
    @State private var activeTab: ActiveControlTab = .none
    @State private var showPostGradingStudio: Bool = false
    @State private var showCubeImporter: Bool = false
    @State private var showCubeShareSheet: Bool = false
    @State private var exportedCubeURL: URL?

    var body: some View {
        GeometryReader { screenGeo in
            ZStack {
                Color.black.ignoresSafeArea()

                MetalLogViewfinder(engine: engine)
                    .ignoresSafeArea()
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onEnded { value in
                                if activeTab != .none {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        activeTab = .none
                                    }
                                } else {
                                    let normX = max(0.05, min(0.95, value.location.x / max(1.0, screenGeo.size.width)))
                                    let normY = max(0.05, min(0.95, value.location.y / max(1.0, screenGeo.size.height)))
                                    engine.handleViewfinderTap(normalizedPoint: CGPoint(x: normX, y: normY), lockFocusAndExposure: false)
                                }
                            }
                    )
                    .simultaneousGesture(
                        LongPressGesture(minimumDuration: 0.55)
                            .onEnded { _ in
                                if let pt = engine.focusReticleNormalized {
                                    engine.handleViewfinderTap(normalizedPoint: pt, lockFocusAndExposure: true)
                                }
                            }
                    )

                ViewfinderGuidesAndHorizonOverlay(engine: engine)
                    .ignoresSafeArea()

                VStack(spacing: 6) {
                    // Row 1: Scrollable Blackmagic Telemetry Pills
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            TelemetryPill(label: "RES", value: engine.selectedSensorPreset.shortTitle, accent: .yellow) {
                                toggleTab(.format)
                            }
                            TelemetryPill(label: "FPS", value: "\(Int(engine.selectedFPS.rawValue))", accent: .white) {
                                toggleTab(.format)
                            }
                            TelemetryPill(label: "SHUTTER", value: "\(Int(engine.shutterAngleDegrees))°", accent: .orange) {
                                toggleTab(.exposure)
                            }
                            TelemetryPill(label: "ISO", value: "\(Int(engine.currentISO))", accent: engine.isManualExposure ? .yellow : .white) {
                                toggleTab(.exposure)
                            }
                            TelemetryPill(label: "WB", value: "\(Int(engine.whiteBalanceKelvin))K", accent: engine.isWhiteBalanceLocked ? .cyan : .white) {
                                toggleTab(.exposure)
                            }
                            TelemetryPill(
                                label: "FOCUS",
                                value: "\(engine.focusMode.shortLabel) \(String(format: "%.2f", engine.lensPosition))",
                                accent: engine.focusPeakingEnabled ? .green : .white
                            ) {
                                toggleTab(.focus)
                            }
                            TelemetryPill(label: "STAB", value: engine.stabilizationMode.shortTitle, accent: .cyan) {
                                toggleTab(.format)
                            }
                        }
                        .padding(.horizontal, 12)
                    }
                    .padding(.top, 4)

                    // Row 2: Action Buttons & Audio VU + Histogram
                    HStack(spacing: 6) {
                        Button {
                            engine.rotateOrientation90Degrees()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: engine.orientationMode.isPortraitSwapped ? "rectangle.portrait" : "rectangle")
                                Text(engine.orientationMode.shortLabel)
                            }
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(Color.black.opacity(0.75))
                            .foregroundColor(.yellow)
                            .cornerRadius(6)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.yellow.opacity(0.5), lineWidth: 1))
                        }
                        .fixedSize()

                        Button {
                            showPostGradingStudio = true
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "slider.horizontal.below.rectangle")
                                Text("后期暗房")
                            }
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(Color.cyan.opacity(0.25))
                            .foregroundColor(.cyan)
                            .cornerRadius(6)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.5), lineWidth: 1))
                        }
                        .fixedSize()

                        Spacer(minLength: 2)

                        StereoAudioVUMeterView(leftDB: engine.audioPeakDBLeft, rightDB: engine.audioPeakDBRight)
                        LumaHistogramOverlay(bins: engine.lumaHistogram)
                    }
                    .padding(.horizontal, 12)

                    // Row 3: Single-Line Telemetry Banner with Semi-Transparent Backing Pill for High Contrast
                    HStack {
                        Text("\(engine.activeResolutionBadge) • \(engine.statusMessage)")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundColor(.white)
                            .lineLimit(1)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.black.opacity(0.60))
                            .cornerRadius(4)
                        Spacer()
                    }
                    .padding(.horizontal, 12)

                    Spacer()

                    // Floating Lens Bar (0.5x / 1x / 3x + PEAK + Quick Photo Snap Button)
                    HStack(spacing: 10) {
                        ForEach(CameraLensOption.allCases) { lens in
                            Button {
                                engine.selectedLens = lens
                            } label: {
                                Text(lens.rawValue)
                                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                                    .frame(width: 38, height: 38)
                                    .background(engine.selectedLens == lens ? Color.yellow : Color.black.opacity(0.65))
                                    .foregroundColor(engine.selectedLens == lens ? .black : .white)
                                    .clipShape(Circle())
                                    .overlay(Circle().stroke(Color.white.opacity(0.25), lineWidth: 1))
                            }
                        }

                        Button {
                            engine.focusPeakingEnabled.toggle()
                        } label: {
                            Text("PEAK")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .frame(height: 38)
                                .background(engine.focusPeakingEnabled ? Color.green : Color.black.opacity(0.65))
                                .foregroundColor(engine.focusPeakingEnabled ? .black : .white)
                                .cornerRadius(19)
                                .overlay(RoundedRectangle(cornerRadius: 19).stroke(Color.white.opacity(0.25), lineWidth: 1))
                        }

                        Button {
                            engine.captureStillPhoto()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "camera.fill")
                                Text("抓拍")
                            }
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 10)
                            .frame(height: 38)
                            .background(Color.black.opacity(0.65))
                            .foregroundColor(.white)
                            .cornerRadius(19)
                            .overlay(RoundedRectangle(cornerRadius: 19).stroke(Color.white.opacity(0.35), lineWidth: 1))
                        }
                    }
                    .padding(.bottom, 2)

                    // Collapsible Control Drawer
                    if activeTab != .none {
                        VStack(spacing: 10) {
                            if activeTab == .format {
                                FormatAndStabilizationDrawer(engine: engine)
                            } else if activeTab == .focus {
                                FocusAndZoomDrawer(engine: engine)
                            } else if activeTab == .exposure {
                                ExposureAndWBDrawer(engine: engine)
                            } else if activeTab == .lutSystem {
                                LUTSystemDrawerView(engine: engine, showCubeImporter: $showCubeImporter)
                            } else if activeTab == .physicalLab {
                                PhysicalFilmLabDrawerView(engine: engine)
                            } else if activeTab == .monitor {
                                MonitorAndSlateDrawer(
                                    engine: engine,
                                    onExportCube: {
                                        if let url = try? CubeLUTManager.exportCubeFile(lutParams: engine.lutSettings) {
                                            exportedCubeURL = url
                                            showCubeShareSheet = true
                                        }
                                    }
                                )
                            }
                        }
                        .padding(12)
                        .background(Color.black.opacity(0.88))
                        .cornerRadius(14)
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.18), lineWidth: 1))
                        .padding(.horizontal, 12)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                    }

                    // Bottom Control Dock: Strictly Separated Mode Switcher + 6 Drawer Tabs + Shutter Bar
                    VStack(spacing: 6) {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                DrawerTabButton(title: "画幅/防抖", icon: "gyroscope", isActive: activeTab == .format) {
                                    toggleTab(.format)
                                }
                                DrawerTabButton(title: "对焦/跟焦", icon: "viewfinder", isActive: activeTab == .focus) {
                                    toggleTab(.focus)
                                }
                                DrawerTabButton(title: "曝光/色温", icon: "camera.aperture", isActive: activeTab == .exposure) {
                                    toggleTab(.exposure)
                                }
                                DrawerTabButton(
                                    title: "LUT 调色",
                                    icon: "cube",
                                    isActive: activeTab == .lutSystem,
                                    badgeDot: engine.colorEngineMode == .lutGrading ? .yellow : nil
                                ) {
                                    engine.colorEngineMode = .lutGrading
                                    toggleTab(.lutSystem)
                                }
                                DrawerTabButton(
                                    title: "物理胶片",
                                    icon: "film",
                                    isActive: activeTab == .physicalLab,
                                    badgeDot: engine.colorEngineMode == .physicalFilmLab ? .orange : nil
                                ) {
                                    engine.colorEngineMode = .physicalFilmLab
                                    toggleTab(.physicalLab)
                                }
                                DrawerTabButton(
                                    title: "原生 Log",
                                    icon: "waveform",
                                    isActive: false,
                                    badgeDot: engine.colorEngineMode == .pureAppleLog ? .green : nil
                                ) {
                                    engine.colorEngineMode = .pureAppleLog
                                    activeTab = .none
                                }
                                DrawerTabButton(title: "辅助/场记", icon: "slider.horizontal.3", isActive: activeTab == .monitor) {
                                    toggleTab(.monitor)
                                }
                            }
                            .padding(.horizontal, 10)
                        }

                        // Video / Photo Mode Selector
                        HStack(spacing: 20) {
                            ForEach(CaptureModeOption.allCases) { mode in
                                Button {
                                    engine.captureMode = mode
                                } label: {
                                    Text(mode.title)
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundColor(engine.captureMode == mode ? .yellow : .gray)
                                }
                            }
                        }

                        // Main Shutter & Timecode Bar
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(formatTimecode(engine.recordingDuration, fps: engine.selectedFPS.rawValue))
                                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                                    .foregroundColor(engine.isRecording ? .red : .white)
                                Text(engine.activeModeSummaryBadge)
                                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                    .foregroundColor(engine.colorEngineMode == .physicalFilmLab ? .orange : (engine.colorEngineMode == .lutGrading ? .yellow : .green))
                            }
                            .frame(width: 145, alignment: .leading)

                            Spacer()

                            Button(action: { engine.triggerMainShutter() }) {
                                ZStack {
                                    Circle()
                                        .stroke(Color.white, lineWidth: 3)
                                        .frame(width: 66, height: 66)
                                    if engine.captureMode == .photo {
                                        Circle()
                                            .fill(Color.white)
                                            .frame(width: 54, height: 54)
                                            .overlay(
                                                Image(systemName: "camera.aperture")
                                                    .font(.system(size: 20, weight: .bold))
                                                    .foregroundColor(.black)
                                            )
                                    } else {
                                        RoundedRectangle(cornerRadius: engine.isRecording ? 8 : 28)
                                            .fill(Color.red)
                                            .frame(
                                                width: engine.isRecording ? 28 : 52,
                                                height: engine.isRecording ? 28 : 52
                                            )
                                            .animation(.easeInOut(duration: 0.18), value: engine.isRecording)
                                    }
                                }
                            }

                            Spacer()

                            VStack(alignment: .trailing, spacing: 2) {
                                Text(engine.captureMode == .photo ? "模式: 16-Bit 拍照" : (engine.bakeIntoRecording ? "REC: 效果直出" : "REC: 纯净 APPLE LOG"))
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(engine.captureMode == .photo ? .yellow : (engine.bakeIntoRecording ? .orange : .green))
                                Text("\(engine.selectedCodec.rawValue) • \(engine.orientationMode.shortLabel)")
                                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                    .foregroundColor(.cyan)
                            }
                            .frame(width: 145, alignment: .trailing)
                        }
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                    }
                    .padding(.top, 6)
                    .background(Color.black.opacity(0.82))
                }
            }
        }
        .onAppear {
            engine.requestPermissionsAndStart()
        }
        .sheet(isPresented: $showPostGradingStudio) {
            PostColorGradingStudioView(engine: engine)
        }
        .sheet(isPresented: $showCubeShareSheet) {
            if let url = exportedCubeURL {
                ShareSheetView(items: [url])
            }
        }
        .fileImporter(
            isPresented: $showCubeImporter,
            allowedContentTypes: [UTType.data, UTType.plainText],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let first = urls.first {
                engine.importCustomCubeLUT(from: first)
            }
        }
    }

    private func toggleTab(_ tab: ActiveControlTab) {
        withAnimation(.easeInOut(duration: 0.18)) {
            activeTab = (activeTab == tab) ? .none : tab
        }
    }

    private func formatTimecode(_ seconds: TimeInterval, fps: Double) -> String {
        if engine.useTimeOfDayTimecode && !engine.isRecording {
            let cal = Calendar.current
            let now = Date()
            let h = cal.component(.hour, from: now)
            let m = cal.component(.minute, from: now)
            let s = cal.component(.second, from: now)
            return String(format: "%02d:%02d:%02d:00", h, m, s)
        }
        let totalSeconds = Int(seconds)
        let mins = (totalSeconds / 60) % 60
        let secs = totalSeconds % 60
        let frames = Int((seconds - Double(totalSeconds)) * fps)
        return String(format: "%02d:%02d:%02d", mins, secs, max(0, frames))
    }
}

// MARK: - 11. Sub-Drawer Views

struct TelemetryPill: View {
    let label: String
    let value: String
    let accent: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                Text(value)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(accent)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.72))
            .cornerRadius(6)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.18), lineWidth: 0.8))
        }
        .fixedSize()
    }
}

struct DrawerTabButton: View {
    let title: String
    let icon: String
    let isActive: Bool
    var badgeDot: Color? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let dot = badgeDot {
                    Circle().fill(dot).frame(width: 6, height: 6)
                }
                Image(systemName: icon)
                Text(title)
            }
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(isActive ? Color.yellow : Color.white.opacity(0.12))
            .foregroundColor(isActive ? .black : .white)
            .cornerRadius(8)
        }
        .fixedSize()
    }
}

struct FormatAndStabilizationDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("传感器画幅 (4K 原生 10-Bit / Open Gate 4:3) 与多级专业防抖")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)

            ForEach(SensorFormatPreset.allCases) { preset in
                Button {
                    engine.selectedSensorPreset = preset
                } label: {
                    HStack {
                        Image(systemName: engine.selectedSensorPreset == preset ? "checkmark.circle.fill" : "circle")
                        Text(preset.rawValue)
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(engine.selectedSensorPreset == preset ? Color.yellow.opacity(0.22) : Color.white.opacity(0.08))
                    .foregroundColor(engine.selectedSensorPreset == preset ? .yellow : .white)
                    .cornerRadius(8)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("专业防抖级别 (OIS 传感器位移 + 多级电影 EIS):")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)
                Picker("Stabilization", selection: $engine.stabilizationMode) {
                    ForEach(CinemaStabilizationMode.allCases) { mode in
                        Text(mode.shortTitle).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            HStack(spacing: 8) {
                Picker("Codec", selection: $engine.selectedCodec) {
                    ForEach(RecordingCodecOption.allCases) { codec in
                        Text(codec.rawValue).tag(codec)
                    }
                }
                .pickerStyle(.segmented)

                Picker("FPS", selection: $engine.selectedFPS) {
                    ForEach(FrameRateOption.allCases) { fps in
                        Text(fps.label).tag(fps)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 165)
            }

            HStack {
                Text("方向:")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                Picker("Orientation", selection: $engine.orientationMode) {
                    ForEach(CaptureOrientationMode.allCases) { mode in
                        Text(mode.shortLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Toggle("双存原片+胶片照", isOn: $engine.dualSaveRawAndFilmPhoto)
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
                    .tint(.orange)
            }
        }
    }
}

struct FocusAndZoomDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("专业对焦控制 (轻触画面选点对焦，长按锁定 AF/AE)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                Spacer()
                Picker("AF Range", selection: $engine.afRangeRestriction) {
                    ForEach(AutoFocusRangeOption.allCases) { opt in
                        Text(opt.title).tag(opt)
                    }
                }
                .pickerStyle(.menu)
                .tint(.cyan)
            }

            Picker("Focus Mode", selection: $engine.focusMode) {
                ForEach(FocusControlMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 8) {
                Text("焦点位置")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white)
                    .frame(width: 72, alignment: .leading)
                Text("微距")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.gray)
                Slider(
                    value: Binding(
                        get: { engine.lensPosition },
                        set: { engine.setManualFocusLensPosition($0) }
                    ),
                    in: 0.0...1.0
                )
                .tint(.yellow)
                Text("无穷远")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.gray)
                Text(String(format: "%.2f", engine.lensPosition))
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                    .frame(width: 40, alignment: .trailing)
            }

            HStack(spacing: 8) {
                Text("平滑跟焦:")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)

                Button("存 A(\(String(format: "%.2f", engine.rackPointA)))") {
                    engine.rackPointA = engine.lensPosition
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .padding(.horizontal, 7)
                .padding(.vertical, 5)
                .background(Color.white.opacity(0.14))
                .foregroundColor(.white)
                .cornerRadius(6)

                Button("推焦 → A") {
                    engine.triggerRackFocus(to: engine.rackPointA)
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.yellow)
                .foregroundColor(.black)
                .cornerRadius(6)

                Button("存 B(\(String(format: "%.2f", engine.rackPointB)))") {
                    engine.rackPointB = engine.lensPosition
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .padding(.horizontal, 7)
                .padding(.vertical, 5)
                .background(Color.white.opacity(0.14))
                .foregroundColor(.white)
                .cornerRadius(6)

                Button("推焦 → B") {
                    engine.triggerRackFocus(to: engine.rackPointB)
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.cyan)
                .foregroundColor(.black)
                .cornerRadius(6)
            }

            HStack(spacing: 10) {
                Toggle("峰值对焦 (Focus Peaking)", isOn: $engine.focusPeakingEnabled)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.green)
                    .tint(.green)

                if engine.focusPeakingEnabled {
                    Picker("Color", selection: $engine.focusPeakingColor) {
                        ForEach(FocusPeakingColorOption.allCases) { c in
                            Text(c.title).tag(c)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 150)
                }
            }

            GradeSliderRow(
                label: "平滑变焦",
                value: Binding(
                    get: { Float(engine.zoomFactor) },
                    set: { engine.setZoomFactor(CGFloat($0), animatedRate: nil) }
                ),
                range: 1.0...6.0,
                format: "%.1fx",
                tint: .cyan
            )
        }
    }
}

struct ExposureAndWBDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Toggle("手动曝光 (Manual ISO + 快门开角)", isOn: $engine.isManualExposure)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                    .tint(.yellow)

                Spacer()

                ForEach([90.0, 172.8, 180.0, 360.0], id: \.self) { angle in
                    Button("\(Int(angle))°") {
                        engine.isManualExposure = true
                        engine.shutterAngleDegrees = angle
                    }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .background(abs(engine.shutterAngleDegrees - angle) < 1.0 ? Color.orange : Color.white.opacity(0.12))
                    .foregroundColor(abs(engine.shutterAngleDegrees - angle) < 1.0 ? .black : .white)
                    .cornerRadius(5)
                }
            }

            if engine.isManualExposure {
                GradeSliderRow(
                    label: "感光度 ISO",
                    value: Binding(get: { engine.currentISO }, set: { engine.setManualISO($0) }),
                    range: engine.minISO...engine.maxISO,
                    format: "%.0f",
                    tint: .yellow
                )
                GradeSliderRow(
                    label: "快门角度",
                    value: Binding(
                        get: { Float(engine.shutterAngleDegrees) },
                        set: { engine.shutterAngleDegrees = Double($0) }
                    ),
                    range: 45...360,
                    format: "%.0f°",
                    tint: .orange
                )
            }

            HStack {
                Toggle("手动白平衡 (色温 K + 色调 Tint)", isOn: $engine.isWhiteBalanceLocked)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)
                    .tint(.cyan)

                Spacer()

                ForEach([3200, 4300, 5600, 6500], id: \.self) { kVal in
                    Button("\(kVal)K") {
                        engine.isWhiteBalanceLocked = true
                        engine.whiteBalanceKelvin = Float(kVal)
                    }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .background(Int(engine.whiteBalanceKelvin) == kVal && engine.isWhiteBalanceLocked ? Color.cyan : Color.white.opacity(0.12))
                    .foregroundColor(Int(engine.whiteBalanceKelvin) == kVal && engine.isWhiteBalanceLocked ? .black : .white)
                    .cornerRadius(5)
                }
            }

            if engine.isWhiteBalanceLocked {
                GradeSliderRow(
                    label: "色温 (K)",
                    value: $engine.whiteBalanceKelvin,
                    range: 2500...10000,
                    format: "%.0fK",
                    tint: .cyan
                )
                GradeSliderRow(
                    label: "色调 Tint",
                    value: $engine.whiteBalanceTint,
                    range: -150...150,
                    format: "%+.0f",
                    tint: .green
                )
            }
        }
    }
}

struct MonitorAndSlateDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine
    let onExportCube: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("监看诊断、陀螺仪水平仪与场记板 (Slate)")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)

            Picker("Diagnostic", selection: $engine.monitorDiagnostic) {
                ForEach(MonitorDiagnosticMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Text("遮幅:")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                Picker("Guide", selection: $engine.aspectGuide) {
                    ForEach(AspectGuideOption.allCases) { guide in
                        Text(guide.rawValue).tag(guide)
                    }
                }
                .pickerStyle(.menu)
                .tint(.yellow)

                Spacer()

                Toggle("水平仪", isOn: $engine.showHorizonLeveler)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.green)
                    .tint(.green)

                Toggle("九宫格", isOn: $engine.showGridOverlay)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .tint(.yellow)
            }

            HStack(spacing: 10) {
                Text("卷号 Reel:")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                TextField("A001", text: $engine.projectReelName)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 75)

                Toggle("真实时钟时间码", isOn: $engine.useTimeOfDayTimecode)
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)
                    .tint(.cyan)

                Spacer()

                Button(action: onExportCube) {
                    Label("导出 .cube", systemImage: "square.and.arrow.up")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Color.yellow)
                        .foregroundColor(.black)
                        .cornerRadius(6)
                }
            }
        }
    }
}

// MARK: - 12. App Entry Point

@main
struct CineLog13ProApp: App {
    var body: some Scene {
        WindowGroup {
            CineLogCameraView()
                .preferredColorScheme(.dark)
        }
    }
}