//
//  iPhone13ProMaxLogCameraApp.swift
//  CineLog13Pro v5.0 (Dual Independent Color Engines: LUT Grading System + Direct-Photon Physical Film Lab)
//  10-Bit Apple Log, 4:3 Open Gate, 10 Built-in Cinema LUTs + .cube Tetrahedral Engine, 12 Direct-Photon Physical Film Stocks,
//  High-Speed Non-Blocking Metal Pipeline (60fps Zero-Stutter), Still Photo Capture & Multi-Stage Stabilization
//
//  Target: iOS 16.0+ (Optimized for iPhone 13 Pro Max A15 Bionic & compatible with iPhone 15 Pro+)
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

// MARK: - 1. Embedded High-Performance Metal Compute & Render Shaders
// Architectural Highlights in v5.0:
// 1. Two Independent Color Engines:
//    - Engine A (LUT Grading System): Sensor -> Linear -> Apple Log -> Primary Wheels -> 10 Built-in Cinema LUTs / Custom .cube 3D LUT (Tetrahedral)
//    - Engine B (Physical Film Lab):  Sensor -> Direct Scene-Linear Photon Flux (bypasses Log!) -> Lens Physics -> Halation -> Base-10 H&D Curve -> Darkroom Paper Print
// 2. High-Speed Non-Blocking Pipeline:
//    - Only writes recordOutTex when recording or snapping a still photo (saves 50% GPU memory bandwidth during preview!)
//    - Fast 4-tap Poisson disk halation & conditional de-banding for locked, stutter-free 4K/OpenGate performance on A15 Bionic.

enum MetalShaderLibrarySource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct ColorGradeUniforms {
        int isNativeAppleLog;       // 1 = hardware native Apple Log, 0 = HLG/SDR sensor input
        int isFullRangeYCbCr;       // 1 = Full Range ('420f'), 0 = Video Range ('x422', 'x420', '420v')
        int isSDRSensorInput;       // 1 = 8-bit SDR P3/709 sensor fallback, 0 = 10-bit HLG BT.2020
        int rotationMode;           // 0 = Portrait (90 CW), 1 = Landscape Right (0 deg), 2 = Landscape Left (180 deg), 3 = Portrait UpsideDown (270 CW)
        int mirrorHorizontal;       // 1 = Mirror X, 0 = Normal
        int writeRecordBuffer;      // 1 = Actively recording video or capturing photo, 0 = Preview only (saves 50% GPU bandwidth!)

        int activeColorSystem;      // 0 = LUT Color Grading System (Log-based), 1 = Physical Film Lab System (Direct Scene-Linear Photon-based)
        int lutPresetID;            // 0 = Raw Apple Log, 1 = Official Rec.709, 2 = ARRI Alexa 709, 3 = Kodak 2383 D65, 4 = Kodak 2383 Warm D55,
                                    // 5 = Fuji 3513DI, 6 = Teal & Orange, 7 = Bleach Bypass, 8 = Cyber Nocturne, 9 = Custom .cube 3D LUT
        int lutInputSpace;          // 0 = Custom .cube expects Apple Log [0,1], 1 = Custom .cube expects Rec.709 [0,1]
        int customLUTSize;          // Dimension N of loaded 3D .cube LUT (17, 33, 64, 65)
        int bakeGradeIntoRecording; // 1 = Bake active LUT/Film into recorded video, 0 = Record pure Apple Log & only preview LUT/Film
        int monitorDiagnosticMode;  // 0 = Normal, 1 = False Color IRE, 2 = Zebra (>94% IRE)
        int hasCustom3DLUT;         // 1 = Custom .cube 3D texture bound at texture(4)

        int focusPeakingEnabled;    // 1 = Highlight sharp edges in viewfinder via 3x3 Sobel filter
        int focusPeakingColorIndex; // 0 = Green, 1 = Red, 2 = Yellow, 3 = Cyan
        float focusPeakingThreshold;// Edge magnitude threshold (0.04 to 0.25)

        // LUT System & Primary Color Grading Controls
        float exposureEV;           // -3.0 to +3.0 EV
        float contrast;             // 0.5 to 1.8 (1.0 = neutral)
        float pivot;                // Contrast pivot in Log space (default 0.488 = 18% gray)
        float highlightRollOff;     // 0.0 to 1.0 shoulder compression
        float shadowLift;           // -0.25 to +0.25 shadow toe adjustment
        float saturation;           // 0.0 to 2.0
        float vibrance;             // -1.0 to +1.0
        float temperatureOffset;    // -1.0 (Cool) to +1.0 (Warm)
        float tintOffset;           // -1.0 (Green) to +1.0 (Magenta)
        float lutIntensity;         // 0.0 to 1.0 blend for LUT / Film stock

        // Physical Film Lab Parameters (Direct Scene-Linear Optics, Emulsion & Darkroom Chemistry)
        int filmStockID;            // 0..11 (12 classic physics-modeled stocks: Vision3 500T, 250D, Cinestill 800T, Portra 400, Gold 200, Ektar 100,
                                    //        Fuji Pro 400H, Superia 400, Velvia 50, Provia 100F, Kodachrome 64, Ilford HP5 400 B&W)
        float pushPullStops;        // Darkroom chemical development time (-1.5 Pull to +2.0 Push stops)
        float halationIntensity;    // Red-layer film base backscatter glow (0.0 to 1.0)
        float halationRadius;       // Spatial scatter radius (1.0 to 10.0)
        float emulsionBloom;        // Silver halide intra-layer Mie optical diffusion (0.0 to 1.0)
        float lensVignette;         // Physical cos^4(theta) optical illumination falloff in linear light (0.0 to 1.0)
        float chromaticAberration;  // Radial wavelength dispersion R/B magnification shift (0.0 to 1.0)
        float printPaperContrast;   // Darkroom enlarger paper grade (0.6 = Grade 1 to 1.5 = Grade 4)
        float printWarmthSplit;     // Darkroom dichroic head split-toning (-1.0 Cool Selenium to +1.0 Warm Amber/Sepia)
        float filmGrain;            // Physical Selwyn-Poisson silver halide grain density (0.0 to 1.0)
        float grainSize;            // 1.0 = 135 (35mm), 0.6 = 120 Medium Format, 1.6 = 16mm Cinema
        float debandingStrength;    // Spatial bilateral gradient smoothing for 8-bit/LUT banding (0.0 to 1.0)
        float frameTime;            // Time seed for organic grain & zebra animation

        float4 liftRGB;             // Shadow RGB offset (-0.2 to +0.2)
        float4 gammaRGB;            // Midtone RGB power/offset (-0.3 to +0.3)
        float4 gainRGB;             // Highlight RGB multiplier (0.5 to 1.5, default 1.0)
    };

    inline float hash12(float2 p) {
        float3 p3 = fract(float3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    }

    // --- 1. Fast YCbCr to Normalized RGB Conversion ---
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

    // Fast 4-Tap Bilateral De-Banding + TPDF Blue-Noise Dither (Runs in <0.4ms, eliminates ceiling/sky rings)
    inline void sampleDeBandedYCbCr(
        texture2d<float, access::read> lumaPlane,
        texture2d<float, access::read> chromaPlane,
        uint2 lumaCoord,
        uint srcW,
        uint srcH,
        float debandStrength,
        float2 ditherSeed,
        thread float& outY,
        thread float2& outCbCr
    ) {
        uint cw = chromaPlane.get_width();
        uint ch = chromaPlane.get_height();
        uint2 chromaCenter = uint2(lumaCoord.x * cw / srcW, lumaCoord.y * ch / srcH);

        float centerY = lumaPlane.read(lumaCoord).r;
        float2 centerC = chromaPlane.read(chromaCenter).rg;

        if (debandStrength <= 0.01) {
            outY = centerY;
            outCbCr = centerC;
            return;
        }

        const int2 offsets[4] = {
            int2( 8,  4), int2(-8, -4), int2(-4,  8), int2( 4, -8)
        };

        float sumY = centerY;
        float2 sumC = centerC;
        float wSumY = 1.0;
        float wSumC = 1.0;
        float invVarY = 1.0 / max(0.000001, (0.020 * debandStrength) * (0.020 * debandStrength));
        float invVarC = 1.0 / max(0.000001, (0.025 * debandStrength) * (0.025 * debandStrength));

        for (int i = 0; i < 4; ++i) {
            int nx = clamp(int(lumaCoord.x) + offsets[i].x, 0, int(srcW) - 1);
            int ny = clamp(int(lumaCoord.y) + offsets[i].y, 0, int(srcH) - 1);
            float sY = lumaPlane.read(uint2(nx, ny)).r;
            float dy = sY - centerY;
            float wy = exp(-(dy * dy) * invVarY);
            sumY += sY * wy;
            wSumY += wy;

            uint2 nC = uint2(uint(nx) * cw / srcW, uint(ny) * ch / srcH);
            float2 sC = chromaPlane.read(nC).rg;
            float2 dc = sC - centerC;
            float wc = exp(-dot(dc, dc) * invVarC);
            sumC += sC * wc;
            wSumC += wc;
        }

        float dither = (hash12(ditherSeed) - hash12(ditherSeed + 17.3)) * (0.0016 * debandStrength);
        outY = clamp((sumY / wSumY) + dither, 0.0, 1.0);
        outCbCr = clamp((sumC / wSumC) + float2(dither * 0.5), 0.0, 1.0);
    }

    // --- 2. Sensor Signal to Calibrated Scene-Linear Light (18% Gray = 0.18, Diffuse White = 1.0) ---
    // Properly calibrated so HLG_BT2020 and P3/SDR map 18% middle gray to 0.18 and black to 0.0001 (never washed out!)
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

    inline float3 sensorToLinearBT2020(float3 encodedRGB, int isSDR, float evBias, float rollOff) {
        float3 sceneLinear;
        if (isSDR == 1) {
            float3 v = clamp(encodedRGB, 0.0, 0.998);
            float3 linP3 = pow(v, float3(2.2));
            float3 knee = max(float3(0.0), v - 0.75) / 0.25;
            linP3 *= (1.0 + 1.4 * knee * knee);
            sceneLinear.r = 0.753833 * linP3.r + 0.198597 * linP3.g + 0.047570 * linP3.b;
            sceneLinear.g = 0.045744 * linP3.r + 0.941777 * linP3.g + 0.012479 * linP3.b;
            sceneLinear.b = -0.001210 * linP3.r + 0.017602 * linP3.g + 0.983608 * linP3.b;
            sceneLinear = max(sceneLinear, 0.0) * exp2(evBias);
        } else {
            // In HLG_BT2020, 75% HLG signal (e'=0.75) is 100% diffuse white (hlgInverseOETF(0.75) = 0.265).
            // Scaling by 3.77 maps HLG diffuse white (0.75) to 1.0 in scene-linear light and 18% gray (e'=0.38) to 0.18!
            // Subtract tiny black pedestal so sensor dark current maps to true 0.0 black.
            float3 rawLin = float3(
                hlgInverseOETFChannel(encodedRGB.r),
                hlgInverseOETFChannel(encodedRGB.g),
                hlgInverseOETFChannel(encodedRGB.b)
            );
            sceneLinear = max(float3(0.0), (rawLin - 0.0012) * 3.77) * exp2(evBias);
        }

        if (rollOff > 0.001) {
            float3 shoulder = sceneLinear / (1.0 + sceneLinear * (0.25 * rollOff));
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

    // --- 4. Tetrahedral 3D LUT Interpolation (Zero Banding) ---
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

    // --- 5. SYSTEM A: LUT Color Grading Engine (10 Built-in Cinema LUTs + Custom .cube LUT) ---
    inline float3 evaluateLUTSystem(
        float3 appleLogIn,
        constant ColorGradeUniforms& u,
        texture3d<float, access::sample> customLUT
    ) {
        float3 logColor = appleLogIn;

        // Primary Log Controls
        logColor.r += u.temperatureOffset * 0.055 + u.tintOffset * 0.028;
        logColor.g -= u.tintOffset * 0.045;
        logColor.b -= u.temperatureOffset * 0.055 - u.tintOffset * 0.028;

        float3 lifted = logColor + u.liftRGB.xyz * (1.0 - logColor);
        float3 gained = lifted * u.gainRGB.xyz;
        float3 gammaPower = max(float3(0.25), 1.0 - u.gammaRGB.xyz * 0.65);
        logColor = pow(clamp(gained, 0.0, 1.0), gammaPower);

        if (abs(u.shadowLift) > 0.001) {
            float shadowWeight = clamp(1.0 - (logColor.g / 0.5), 0.0, 1.0);
            logColor += u.shadowLift * shadowWeight * shadowWeight * 0.25;
        }
        logColor = clamp((logColor - u.pivot) * u.contrast + u.pivot, 0.0, 1.0);

        // Preset 0: Raw Apple Log
        if (u.lutPresetID == 0) {
            float luma = dot(logColor, float3(0.2627, 0.6780, 0.0593));
            return clamp(mix(float3(luma), logColor, u.saturation), 0.0, 1.0);
        }

        float3 lin2020 = appleLogToLinear(logColor);
        float3 lin709 = linearBT2020ToLinear709(lin2020);

        // ACES / Official Rec.709 Base Display Transform (Rich true blacks & smooth highlight shoulder)
        float3 x = max(float3(0.0), lin709 - 0.0015) * 1.45;
        float3 rec709 = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
        rec709 = pow(rec709, float3(1.0 / 2.2));
        // Contrast S-curve polish so Rec.709 has deep blacks and crisp midtones
        rec709 = mix(rec709, smoothstep(0.015, 0.99, rec709), 0.35);

        float3 styled = rec709;
        float luma = dot(rec709, float3(0.2126, 0.7152, 0.0722));

        if (u.lutPresetID == 2) {
            // ARRI Alexa LogC -> K1S1 Rec.709 Emulation: Creamy highlight roll-off, warm organic skin tones, dense blacks
            float3 arri = pow(clamp(lin709 / (lin709 + 0.24), 0.0, 1.0), float3(1.0 / 2.05));
            arri = smoothstep(0.02, 0.985, arri);
            arri.r = mix(arri.r, arri.r * 1.03, luma);
            arri.g = mix(arri.g, arri.g * 1.01, luma);
            styled = clamp(arri, 0.0, 1.0);
        } else if (u.lutPresetID == 3) {
            // Kodak 2383 Print LUT (D65 Daylight): Deep theater print blacks, punchy S-curve, cyan-blue shadows
            float3 sCurve = smoothstep(0.03, 0.97, rec709);
            float3 shadowSplit = float3(-0.04, 0.008, 0.052) * pow(1.0 - luma, 1.5);
            float3 hiSplit     = float3(0.025, 0.018, -0.025) * pow(luma, 1.2);
            styled = clamp(mix(rec709, sCurve, 0.65) + shadowSplit + hiSplit, 0.0, 1.0);
        } else if (u.lutPresetID == 4) {
            // Kodak 2383 Print LUT (D55 Warm Gold): Classic golden-hour print stock
            float3 sCurve = smoothstep(0.03, 0.97, rec709);
            float3 warmHi = float3(0.065, 0.038, -0.048) * pow(luma, 0.9);
            float3 tealSh = float3(-0.025, 0.012, 0.035) * pow(1.0 - luma, 1.4);
            styled = clamp(mix(rec709, sCurve, 0.60) + warmHi + tealSh, 0.0, 1.0);
        } else if (u.lutPresetID == 5) {
            // Fujifilm 3513DI Print LUT: Crisp emerald greens, cool cyan mid-shadows, clean neutral highs
            float3 sCurve = smoothstep(0.025, 0.975, rec709);
            sCurve.g = clamp(sCurve.g * 1.035, 0.0, 1.0);
            sCurve.b = clamp(sCurve.b * 1.025 + 0.02 * (1.0 - luma), 0.0, 1.0);
            styled = mix(rec709, sCurve, 0.60);
        } else if (u.lutPresetID == 6) {
            // Teal & Orange Master LUT
            float3 sCurve = smoothstep(0.025, 0.975, rec709);
            float3 sh = float3(-0.065, 0.018, 0.085) * pow(1.0 - luma, 1.35);
            float3 hi = float3(0.075, 0.028, -0.055) * pow(luma, 0.85);
            styled = clamp(mix(rec709, sCurve, 0.55) + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 7) {
            // Bleach Bypass Silver Retention LUT
            float3 desat = mix(float3(luma), rec709, 0.38);
            styled = smoothstep(0.05, 0.95, desat);
        } else if (u.lutPresetID == 8) {
            // Cyber Neon Nocturne LUT
            float3 sCurve = smoothstep(0.03, 0.97, rec709);
            float3 sh = float3(0.02, -0.03, 0.09) * (1.0 - luma);
            float3 hi = float3(-0.02, 0.04, 0.06) * luma;
            styled = clamp(sCurve + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 9 && u.hasCustom3DLUT == 1) {
            float3 lutSource = (u.lutInputSpace == 0) ? logColor : rec709;
            styled = sample3DLUTTetrahedral(customLUT, lutSource, u.customLUTSize);
        }

        float3 result = mix(rec709, styled, clamp(u.lutIntensity, 0.0, 1.0));
        float finalLuma = dot(result, float3(0.2126, 0.7152, 0.0722));
        float maxC = max(result.r, max(result.g, result.b));
        float minC = min(result.r, min(result.g, result.b));
        float vibranceFactor = 1.0 + u.vibrance * (1.0 - (maxC - minC));
        return clamp(mix(float3(finalLuma), result, max(0.0, u.saturation * vibranceFactor)), 0.0, 1.0);
    }

    // --- 6. SYSTEM B: Direct-Photon Physical Optics, Emulsion & Darkroom Chemistry Film Simulation ---
    // Operates directly on Calibrated Scene-Linear Light (NEVER passes through Apple Log!)
    // Uses true Base-10 Hurter-Driffield sensitometry (10^x = exp2(3.321928 * x)) and anchored Print Paper Dmax blacks!
    inline float3 evaluatePhysicalFilmLab(
        float3 directSceneLinear709,
        float3 halationScatterLinear,
        constant ColorGradeUniforms& u,
        uint2 pixelCoord,
        uint2 frameSize
    ) {
        // 1. Optical Lens Vignetting (cos^4(theta) illumination falloff in linear photon space)
        float2 normUV = (float2(pixelCoord) / float2(max(uint2(1, 1), frameSize))) * 2.0 - 1.0;
        float rSq = dot(normUV, normUV) * 0.45;
        float cos4Falloff = pow(1.0 / sqrt(1.0 + rSq), 4.0 * u.lensVignette);

        // Apply White Balance Temperature/Tint as Optical Dichroic Glass Filter in Linear Light
        float3 opticalFilter = float3(
            1.0 + u.temperatureOffset * 0.18 + u.tintOffset * 0.08,
            1.0 - u.tintOffset * 0.12,
            1.0 - u.temperatureOffset * 0.18 + u.tintOffset * 0.08
        );

        // 2. Inject Physical Film Base Red Backscatter (Halation) & Silver Halide Mie Bloom in Linear Photon Space
        float stockHalationBoost = (u.filmStockID == 2) ? 1.65 : 1.0; // Cinestill 800T (Remjet-free) has stronger red halation
        float3 halationWeights = float3(1.0, 0.24, 0.03) * (u.halationIntensity * stockHalationBoost * 1.15);
        float3 bloomWeights    = float3(0.40, 0.40, 0.40) * (u.emulsionBloom * 0.55);

        float3 linearPhotons = directSceneLinear709 * opticalFilter * cos4Falloff + halationScatterLinear * (halationWeights + bloomWeights);
        linearPhotons = max(linearPhotons, 0.00008);

        // 3. Emulsion Spectral Couplers (DIR Inter-Layer Matrix), Sensitometric Gamma & Color Balance per Classic Stock
        // System Gamma = Negative Gamma * Print Paper Gamma (typically ~1.65 - 2.05 for punchy, authentic film contrast!)
        float3 sysGamma = float3(1.68, 1.70, 1.72);
        float3 toePower = float3(1.15, 1.15, 1.15);
        float3 shoulderSoftness = float3(0.85, 0.85, 0.88);
        float3 shadowTint = float3(0.0, 0.0, 0.0);
        float3 midTint    = float3(0.0, 0.0, 0.0);
        float3 hiTint     = float3(0.0, 0.0, 0.0);
        float stockSat    = 1.05;

        // DIR Chemical Inter-Layer Coupler Matrix
        float3 dirR = float3( 1.08, -0.05, -0.03);
        float3 dirG = float3(-0.04,  1.07, -0.03);
        float3 dirB = float3(-0.03, -0.05,  1.08);

        if (u.filmStockID == 0) {
            // 0: Kodak Vision3 500T (5219 ECN-2 + 2383 Print): Cool cyan-blue shadows, warm golden highlights, cinema contrast
            sysGamma = float3(1.72, 1.74, 1.78);
            shoulderSoftness = float3(0.80, 0.84, 0.88);
            shadowTint = float3(-0.035, 0.008, 0.048);
            hiTint     = float3( 0.045, 0.025, -0.038);
            dirR = float3( 1.10, -0.06, -0.04);
            dirG = float3(-0.04,  1.07, -0.03);
            dirB = float3(-0.04, -0.05,  1.09);
        } else if (u.filmStockID == 1) {
            // 1: Kodak Vision3 250D (5207 ECN-2 Daylight): Clean neutral shadows, rich natural foliage & skin
            sysGamma = float3(1.70, 1.72, 1.72);
            shoulderSoftness = float3(0.82, 0.85, 0.86);
            shadowTint = float3(-0.015, 0.005, 0.022);
            hiTint     = float3( 0.028, 0.016, -0.022);
        } else if (u.filmStockID == 2) {
            // 2: Cinestill 800T (Remjet-Free Tungsten Cinema): Pronounced red halation, tungsten cool-teal cast & neon glow
            sysGamma = float3(1.76, 1.78, 1.82);
            shadowTint = float3(-0.045, 0.015, 0.062);
            hiTint     = float3( 0.055, 0.020, -0.035);
            stockSat   = 1.12;
        } else if (u.filmStockID == 3) {
            // 3: Kodak Portra 400 (C-41 Portrait): Gentle highlight shoulder, warm golden midtones, luminous skin
            sysGamma = float3(1.58, 1.56, 1.52);
            shoulderSoftness = float3(0.75, 0.78, 0.82);
            midTint    = float3( 0.022, 0.010, -0.018);
            hiTint     = float3( 0.035, 0.022, -0.025);
            stockSat   = 1.04;
        } else if (u.filmStockID == 4) {
            // 4: Kodak Gold 200 (C-41 Vintage Consumer): Punchy contrast, rich amber-gold highlights, warm nostalgic reds
            sysGamma = float3(1.78, 1.74, 1.66);
            shadowTint = float3( 0.012, -0.008, -0.015);
            midTint    = float3( 0.035,  0.018, -0.030);
            hiTint     = float3( 0.055,  0.035, -0.045);
            stockSat   = 1.15;
        } else if (u.filmStockID == 5) {
            // 5: Kodak Ektar 100 (C-41 Ultra-Vivid): High DIR coupler purity, deep blacks, saturated primaries
            sysGamma = float3(1.86, 1.86, 1.84);
            dirR = float3( 1.18, -0.10, -0.08);
            dirG = float3(-0.08,  1.16, -0.08);
            dirB = float3(-0.06, -0.10,  1.16);
            stockSat = 1.22;
        } else if (u.filmStockID == 6) {
            // 6: Fujifilm Pro 400H (C-41 4th Cyan Layer): Airy pastel highlights, emerald greens, cool cyan shadows
            sysGamma = float3(1.56, 1.60, 1.60);
            shadowTint = float3(-0.030, 0.018, 0.028);
            hiTint     = float3( 0.010, 0.022, 0.015);
            dirG = float3(-0.06,  1.12, -0.06);
        } else if (u.filmStockID == 7) {
            // 7: Fujifilm Superia X-TRA 400 (C-41 Street): Crisp contrast, subtle magenta-plum shadows & vivid cyan-greens
            sysGamma = float3(1.75, 1.78, 1.76);
            shadowTint = float3( 0.022, -0.018, 0.025);
            hiTint     = float3(-0.015,  0.032, 0.018);
            stockSat   = 1.12;
        } else if (u.filmStockID == 8) {
            // 8: Fujifilm Velvia 50 (E-6 Slide Reversal): Steep reversal gamma, ultra-deep Dmax blacks, vibrant landscape colors
            sysGamma = float3(2.02, 2.04, 2.02);
            toePower = float3(1.30, 1.30, 1.30);
            dirR = float3( 1.22, -0.12, -0.10);
            dirG = float3(-0.09,  1.20, -0.11);
            dirB = float3(-0.08, -0.12,  1.20);
            stockSat = 1.28;
        } else if (u.filmStockID == 9) {
            // 9: Fujifilm Provia 100F (E-6 Professional Slide): Balanced reversal contrast, faithful tonal separation
            sysGamma = float3(1.85, 1.86, 1.88);
            toePower = float3(1.20, 1.20, 1.20);
            shadowTint = float3(-0.012, 0.004, 0.018);
            stockSat = 1.10;
        } else if (u.filmStockID == 10) {
            // 10: Kodak Kodachrome 64 (K-14 Archival Reversal): Dense blacks, warm archival reds, deep cobalt blues
            sysGamma = float3(1.92, 1.88, 1.80);
            toePower = float3(1.25, 1.25, 1.25);
            shadowTint = float3(-0.020, -0.010, 0.035);
            midTint    = float3( 0.032,  0.012, -0.025);
            hiTint     = float3( 0.045,  0.028, -0.040);
            stockSat   = 1.08;
        }

        // Apply Darkroom Push/Pull Chemical Development Time Modifier:
        float pushExposureGain = exp2(u.pushPullStops * 0.85);
        float pushContrastMult = (1.0 + u.pushPullStops * 0.16) * u.printPaperContrast * u.contrast;
        sysGamma *= pushContrastMult;

        // Apply DIR Inter-Layer Coupler Matrix in Linear Photon Space
        float3 coupledLin;
        coupledLin.r = max(0.00005, dot(linearPhotons, dirR)) * pushExposureGain;
        coupledLin.g = max(0.00005, dot(linearPhotons, dirG)) * pushExposureGain;
        coupledLin.b = max(0.00005, dot(linearPhotons, dirB)) * pushExposureGain;

        // Stock 11: Ilford HP5 Plus 400 Silver Gelatin Panchromatic B&W
        if (u.filmStockID == 11) {
            float filterShift = clamp(u.temperatureOffset, -1.0, 1.0);
            float3 panWeights = normalize(float3(0.30 + 0.20 * filterShift, 0.54, 0.16 - 0.12 * filterShift));
            float mono = max(0.00005, dot(coupledLin, panWeights));
            coupledLin = float3(mono);
            sysGamma = float3(1.82 * pushContrastMult);
        }

        // 4. True Base-10 Hurter-Driffield (H&D) Sensitometric S-Curve (Anchored at 18% Middle Gray H0 = 0.18)
        // deltaLogH = log10(H / 0.18): 0 at middle gray, negative in shadows, positive in highlights
        float3 deltaLogH = log10(coupledLin / 0.18);

        // Apply asymmetric toe (shadow density) and shoulder (highlight roll-off) using Base-10 exponentiation 10^x = exp2(3.321928 * x)
        float3 negArg = -sysGamma * deltaLogH / shoulderSoftness;
        float3 rawCurve = 1.0 / pow(1.0 + exp2(clamp(3.321928 * negArg, -18.0, 18.0)), shoulderSoftness);

        // Anchor Black Point (Dmax at deltaLogH = -1.6) and White Point (Dmin at deltaLogH = +1.5) so shadows are NEVER milky gray!
        float3 blackFloor = 1.0 / pow(1.0 + exp2(3.321928 * (sysGamma * 1.65 / shoulderSoftness)), shoulderSoftness);
        float3 whiteCeil  = 1.0 / pow(1.0 + exp2(-3.321928 * (sysGamma * 1.55 / shoulderSoftness)), shoulderSoftness);
        float3 filmPositive = clamp((rawCurve - blackFloor) / max(float3(0.001), whiteCeil - blackFloor), 0.0, 1.0);
        filmPositive = pow(filmPositive, toePower);

        // Apply 3-Way LGG / Shadow Lift fine-tuning if adjusted by user
        if (abs(u.shadowLift) > 0.001) {
            float shMask = pow(1.0 - filmPositive.g, 1.5);
            filmPositive = clamp(filmPositive + u.shadowLift * shMask * 0.22, 0.0, 1.0);
        }

        // 5. Apply Emulsion Dye-Layer Color Balance (Shadow / Midtone / Highlight Chemical Split + Dichroic Head Toning)
        float luma = dot(filmPositive, float3(0.2126, 0.7152, 0.0722));
        if (u.filmStockID != 11) {
            float shW  = pow(1.0 - luma, 1.6);
            float midW = 4.0 * luma * (1.0 - luma);
            float hiW  = pow(luma, 1.3);
            filmPositive = clamp(filmPositive + shadowTint * shW + midTint * midW + hiTint * hiW, 0.0, 1.0);
        }

        if (abs(u.printWarmthSplit) > 0.001) {
            float3 warmHi = float3(0.042, 0.022, -0.038) * u.printWarmthSplit * luma;
            float3 coolSh = float3(-0.030, 0.006, 0.042) * (-u.printWarmthSplit) * (1.0 - luma);
            filmPositive = clamp(filmPositive + warmHi + coolSh, 0.0, 1.0);
        }

        // 6. Selwyn-Poisson Silver Halide Dye-Cloud Grain Clumping (Proportional to sqrt(D * (Dmax - D)))
        if (u.filmGrain > 0.001) {
            float grainScale = max(0.4, u.grainSize);
            float2 gCoord = floor(float2(pixelCoord) / grainScale) + float2(floor(u.frameTime * 24.0) * 19.1, floor(u.frameTime * 24.0) * 31.7);
            float nR = hash12(gCoord) + hash12(gCoord * 1.7 + 5.3) - 1.0;
            float nG = hash12(gCoord + 11.7) + hash12(gCoord * 1.7 + 19.1) - 1.0;
            float nB = hash12(gCoord + 23.4) + hash12(gCoord * 1.7 + 37.9) - 1.0;
            float3 grainNoise = (u.filmStockID == 11) ? float3(nG) : float3(nR * 0.9, nG * 0.85, nB * 1.15);

            float pushGrainMult = 1.0 + max(0.0, u.pushPullStops) * 0.35;
            float bellEnvelope = sqrt(max(0.0, luma * (1.0 - luma))) * 2.0;
            filmPositive = clamp(filmPositive + grainNoise * u.filmGrain * 0.085 * pushGrainMult * bellEnvelope, 0.0, 1.0);
        }

        // 7. Saturation & Vibrance
        float finalLuma = dot(filmPositive, float3(0.2126, 0.7152, 0.0722));
        if (u.filmStockID == 11) {
            return clamp(filmPositive, 0.0, 1.0);
        }
        float maxC = max(filmPositive.r, max(filmPositive.g, filmPositive.b));
        float minC = min(filmPositive.r, min(filmPositive.g, filmPositive.b));
        float vibranceFactor = 1.0 + u.vibrance * (1.0 - (maxC - minC));
        float totalSat = max(0.0, stockSat * u.saturation * vibranceFactor);
        return clamp(mix(float3(finalLuma), filmPositive, totalSat), 0.0, 1.0);
    }

    // --- 7. False Color IRE Exposure Diagnostic Map ---
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

    // --- 8. Sobel Edge Detection for Focus Peaking ---
    inline float computeSobelEdge(texture2d<float, access::read> lumaPlane, uint2 center, uint maxW, uint maxH) {
        uint x0 = (center.x > 0) ? (center.x - 1) : 0;
        uint x2 = (center.x + 1 < maxW) ? (center.x + 1) : (maxW - 1);
        uint y0 = (center.y > 0) ? (center.y - 1) : 0;
        uint y2 = (center.y + 1 < maxH) ? (center.y + 1) : (maxH - 1);

        float tl = lumaPlane.read(uint2(x0, y0)).r;
        float tc = lumaPlane.read(uint2(center.x, y0)).r;
        float tr = lumaPlane.read(uint2(x2, y0)).r;
        float ml = lumaPlane.read(uint2(x0, center.y)).r;
        float mr = lumaPlane.read(uint2(x2, center.y)).r;
        float bl = lumaPlane.read(uint2(x0, y2)).r;
        float bc = lumaPlane.read(uint2(center.x, y2)).r;
        float br = lumaPlane.read(uint2(x2, y2)).r;

        float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
        float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
        return sqrt(gx * gx + gy * gy);
    }

    // --- 9. Fast 4-Tap Physical Halation & Optical Chromatic Dispersion Sampler ---
    inline float3 sampleHalationAndDispersion(
        texture2d<float, access::read> lumaPlane,
        uint2 lumaCoord,
        uint srcW,
        uint srcH,
        constant ColorGradeUniforms& u,
        thread float& centerYShiftR,
        thread float& centerYShiftB
    ) {
        if (u.activeColorSystem != 1) {
            return float3(0.0);
        }

        if (u.chromaticAberration > 0.01) {
            float2 center = float2(srcW, srcH) * 0.5;
            float2 delta = float2(lumaCoord) - center;
            float2 shift = delta * (0.004 * u.chromaticAberration);
            uint2 rCoord = uint2(clamp(float2(lumaCoord) + shift, float2(0.0), float2(srcW - 1, srcH - 1)));
            uint2 bCoord = uint2(clamp(float2(lumaCoord) - shift, float2(0.0), float2(srcW - 1, srcH - 1)));
            centerYShiftR = lumaPlane.read(rCoord).r;
            centerYShiftB = lumaPlane.read(bCoord).r;
        }

        if (u.halationIntensity <= 0.01 && u.emulsionBloom <= 0.01) {
            return float3(0.0);
        }

        float rad = max(3.0, u.halationRadius * 3.0);
        const float2 dirs[4] = {
            float2( 1.0,  0.5), float2(-1.0, -0.5), float2(-0.5,  1.0), float2( 0.5, -1.0)
        };

        float scatterAccum = 0.0;
        for (int i = 0; i < 4; ++i) {
            float2 pos = clamp(float2(lumaCoord) + dirs[i] * rad, float2(0.0), float2(srcW - 1, srcH - 1));
            float yVal = lumaPlane.read(uint2(pos)).r;
            float hi = max(0.0, yVal - 0.65) * 2.85;
            scatterAccum += hi * hi;
        }
        return float3(scatterAccum * 0.25);
    }

    // --- 10. Main Camera Compute Kernel ---
    kernel void processAppleLogFrame(
        texture2d<float, access::read>   lumaPlane     [[texture(0)]],
        texture2d<float, access::read>   chromaPlane   [[texture(1)]],
        texture2d<float, access::write>  recordOutTex  [[texture(2)]],
        texture2d<float, access::write>  previewOutTex [[texture(3)]],
        texture3d<float, access::sample> customLUTTex  [[texture(4)]],
        constant ColorGradeUniforms&     uniforms      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = previewOutTex.get_width();
        uint outH = previewOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }

        uint srcW = lumaPlane.get_width();
        uint srcH = lumaPlane.get_height();

        uint destX = (uniforms.mirrorHorizontal == 1) ? (outW - 1 - gid.x) : gid.x;
        uint destY = gid.y;

        uint srcX = 0;
        uint srcY = 0;
        if (uniforms.rotationMode == 0) {
            srcX = min(destY, srcW - 1);
            srcY = (srcH - 1) - min(destX, srcH - 1);
        } else if (uniforms.rotationMode == 1) {
            srcX = min(destX, srcW - 1);
            srcY = min(destY, srcH - 1);
        } else if (uniforms.rotationMode == 2) {
            srcX = (srcW - 1) - min(destX, srcW - 1);
            srcY = (srcH - 1) - min(destY, srcH - 1);
        } else {
            srcX = (srcW - 1) - min(destY, srcW - 1);
            srcY = min(destX, srcH - 1);
        }

        uint2 lumaCoord = uint2(srcX, srcY);

        // Run Bilateral De-Banding only when needed (8-bit fallback or custom LUT active) to keep 4K 10-bit ultra-fast
        float effectiveDeband = (uniforms.isSDRSensorInput == 1 || uniforms.lutPresetID == 9) ? uniforms.debandingStrength : (uniforms.debandingStrength * 0.25);
        float ySample = 0.0;
        float2 cbcrSample = float2(0.5);
        sampleDeBandedYCbCr(
            lumaPlane,
            chromaPlane,
            lumaCoord,
            srcW,
            srcH,
            effectiveDeband,
            float2(gid) + uniforms.frameTime * 13.7,
            ySample,
            cbcrSample
        );

        float yShiftR = ySample;
        float yShiftB = ySample;
        float3 halationLinear = sampleHalationAndDispersion(
            lumaPlane, lumaCoord, srcW, srcH, uniforms, yShiftR, yShiftB
        );

        float3 sensorRGB = decodeYCbCrToRGB(ySample, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput);
        if (uniforms.chromaticAberration > 0.01 && uniforms.activeColorSystem == 1) {
            float3 rgbR = decodeYCbCrToRGB(yShiftR, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput);
            float3 rgbB = decodeYCbCrToRGB(yShiftB, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput);
            sensorRGB.r = rgbR.r;
            sensorRGB.b = rgbB.b;
        }

        // Decode sensor to Calibrated Scene-Linear Light AND Apple Log
        float3 sceneLinearBT2020;
        float3 appleLogRGB;
        if (uniforms.isNativeAppleLog == 1) {
            appleLogRGB = sensorRGB;
            sceneLinearBT2020 = appleLogToLinear(sensorRGB) * exp2(uniforms.exposureEV);
        } else {
            sceneLinearBT2020 = sensorToLinearBT2020(sensorRGB, uniforms.isSDRSensorInput, uniforms.exposureEV, uniforms.highlightRollOff);
            appleLogRGB = linearToAppleLog(sceneLinearBT2020);
        }

        // Dispatch to the Active Independent Color Engine:
        // System 0 = LUT Color Grading Engine (operates on Apple Log)
        // System 1 = Physical Film Lab Engine (operates directly on Scene-Linear Photons!)
        float3 gradedRGB;
        if (uniforms.activeColorSystem == 0) {
            gradedRGB = evaluateLUTSystem(appleLogRGB, uniforms, customLUTTex);
        } else {
            float3 directLinear709 = linearBT2020ToLinear709(sceneLinearBT2020);
            gradedRGB = evaluatePhysicalFilmLab(directLinear709, halationLinear, uniforms, gid, uint2(outW, outH));
        }

        // Only write to recordOutTex when recording or capturing a photo (cuts GPU write bandwidth by 50% during preview!)
        if (uniforms.writeRecordBuffer == 1) {
            float3 recordSignal = (uniforms.bakeGradeIntoRecording == 1) ? gradedRGB : appleLogRGB;
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
        constant ColorGradeUniforms&     uniforms     [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = gradedOutTex.get_width();
        uint outH = gradedOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }
        float3 srcLog = sourceTex.read(gid).rgb;
        float3 lin2020 = appleLogToLinear(srcLog) * exp2(uniforms.exposureEV);
        if (uniforms.highlightRollOff > 0.001) {
            float3 shoulder = lin2020 / (1.0 + lin2020 * (0.25 * uniforms.highlightRollOff));
            lin2020 = mix(lin2020, shoulder, uniforms.highlightRollOff);
        }

        float3 graded;
        if (uniforms.activeColorSystem == 0) {
            float3 adjLog = linearToAppleLog(lin2020);
            graded = evaluateLUTSystem(adjLog, uniforms, customLUTTex);
        } else {
            float3 lin709 = linearBT2020ToLinear709(lin2020);
            graded = evaluatePhysicalFilmLab(lin709, float3(0.0), uniforms, gid, uint2(outW, outH));
        }
        gradedOutTex.write(float4(graded, 1.0), gid);
    }

    // --- 12. Render Pipeline Shaders for MTKView ---
    struct VertexOut {
        float4 position [[position]];
        float2 texCoord;
    };

    struct ViewfinderScaleUniforms {
        float2 scale;
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
        const float2 uvs[4] = {
            float2(0.0, 1.0),
            float2(1.0, 1.0),
            float2(0.0, 0.0),
            float2(1.0, 0.0)
        };
        VertexOut out;
        out.position = float4(positions[vertexID] * vf.scale, 0.0, 1.0);
        out.texCoord = uvs[vertexID];
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

// MARK: - 2. Dual Color Engine, Cinema Camera & Stabilization Data Models

enum ActiveColorSystemMode: Int, CaseIterable, Identifiable {
    case lutSystem      = 0 // LUT 调色引擎 (基于 Apple Log)
    case physicalFilmLab = 1 // 物理胶片暗房 (直接基于线性光子通量)

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .lutSystem:       return "LUT 调色系统 (Log)"
        case .physicalFilmLab: return "物理胶片暗房 (Photon)"
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
    case vision3_500T = 0
    case vision3_250D = 1
    case cinestill800T = 2
    case portra400    = 3
    case gold200      = 4
    case ektar100     = 5
    case fujiPro400H  = 6
    case superia400   = 7
    case velvia50     = 8
    case provia100F   = 9
    case kodachrome64 = 10
    case ilfordHP5    = 11

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
        case .cinestill800T: return "去石墨层电影负片 • 强烈片基红橙色光晕 (Halation) + 夜景霓虹"
        case .portra400:     return "C-41 专业人像负片 • DIR 成色剂耦合 + 温润金黄肤色"
        case .gold200:       return "C-41 经典暖调负片 • 复古金黄高光 + 浓郁厚重反差"
        case .ektar100:      return "C-41 超细颗粒风光负片 • 极高成色剂色彩纯度 + 深邃黑位"
        case .fujiPro400H:   return "C-41 四层感光负片 • 第四青敏层翡翠绿 + 空气感粉白高光"
        case .superia400:    return "C-41 经典街拍负片 • 微妙洋红暗部偏移 + 鲜亮青绿"
        case .velvia50:      return "E-6 传奇风光反转片 • Dmax 3.6 极深黑位 + 高陡峭度正像显影"
        case .provia100F:    return "E-6 专业标准反转片 • 精准中性灰阶 + 细腻正片微反差"
        case .kodachrome64:  return "K-14 三道外偶反转片 • 传奇纪实琥珀黄、暖红与深钴蓝"
        case .ilfordHP5:     return "D-76 全色黑白银盐负片 • 扎实卤化银团簇颗粒 + 0-5号相纸反差"
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

struct ColorGradeParameters: Equatable {
    // Independent System Selector: 0 = LUT Grading System (Log), 1 = Physical Film Lab (Direct Photon)
    var activeColorSystem: ActiveColorSystemMode = .physicalFilmLab
    var lutPreset: BuiltInLUTPresetOption = .kodak2383D65
    var lutInputSpace: LUTInputSpaceOption = .appleLog
    var bakeGradeIntoRecording: Bool = false
    var lutIntensity: Float = 1.0

    // Primary Controls
    var exposureEV: Float = 0.0
    var contrast: Float = 1.0
    var pivot: Float = 0.488
    var highlightRollOff: Float = 0.30
    var shadowLift: Float = 0.0
    var saturation: Float = 1.0
    var vibrance: Float = 0.05
    var temperatureOffset: Float = 0.0
    var tintOffset: Float = 0.0
    var debandingStrength: Float = 0.65

    // Physical Film Lab Parameters (Direct-Photon Optics, Emulsion & Darkroom Chemistry)
    var filmStock: PhysicalFilmStockOption = .vision3_500T
    var pushPullStops: Float = 0.0          // -1.5 to +2.0 stops chemical development time
    var halationIntensity: Float = 0.40     // Film base red backscatter glow
    var halationRadius: Float = 4.0         // Scatter radius
    var emulsionBloom: Float = 0.22         // Silver halide optical diffusion
    var lensVignette: Float = 0.28          // Natural cos^4(theta) lens falloff
    var chromaticAberration: Float = 0.15   // Optical wavelength dispersion
    var printPaperContrast: Float = 1.05    // Enlarger print paper contrast grade
    var printWarmthSplit: Float = 0.10      // Dichroic head split-toning
    var filmGrain: Float = 0.24             // Selwyn-Poisson silver grain density
    var grainSize: Float = 1.0              // 1.0 = 35mm, 0.6 = 120 Medium Format, 1.6 = 16mm

    // 3-Way Color Wheels
    var liftR: Float = 0.0
    var liftG: Float = 0.0
    var liftB: Float = 0.0

    var gammaR: Float = 0.0
    var gammaG: Float = 0.0
    var gammaB: Float = 0.0

    var gainR: Float = 1.0
    var gainG: Float = 1.0
    var gainB: Float = 1.0

    var activeSummaryBadge: String {
        switch activeColorSystem {
        case .lutSystem:
            return "LUT: \(lutPreset.title)"
        case .physicalFilmLab:
            return "胶片: \(filmStock.shortName)"
        }
    }

    mutating func selectLUTPreset(_ preset: BuiltInLUTPresetOption) {
        self.activeColorSystem = .lutSystem
        self.lutPreset = preset
    }

    mutating func applyStockDefaults(_ stock: PhysicalFilmStockOption) {
        self.activeColorSystem = .physicalFilmLab
        self.filmStock = stock
        switch stock {
        case .vision3_500T:
            halationIntensity = 0.42; emulsionBloom = 0.24; filmGrain = 0.26; printPaperContrast = 1.06; printWarmthSplit = 0.14
        case .vision3_250D:
            halationIntensity = 0.25; emulsionBloom = 0.18; filmGrain = 0.18; printPaperContrast = 1.04; printWarmthSplit = 0.06
        case .cinestill800T:
            halationIntensity = 0.78; emulsionBloom = 0.35; filmGrain = 0.32; printPaperContrast = 1.10; printWarmthSplit = 0.10
        case .portra400:
            halationIntensity = 0.20; emulsionBloom = 0.22; filmGrain = 0.18; printPaperContrast = 0.98; printWarmthSplit = 0.20
        case .gold200:
            halationIntensity = 0.28; emulsionBloom = 0.25; filmGrain = 0.24; printPaperContrast = 1.08; printWarmthSplit = 0.32
        case .ektar100:
            halationIntensity = 0.14; emulsionBloom = 0.14; filmGrain = 0.10; printPaperContrast = 1.14; printWarmthSplit = 0.08
        case .fujiPro400H:
            halationIntensity = 0.18; emulsionBloom = 0.20; filmGrain = 0.16; printPaperContrast = 0.98; printWarmthSplit = -0.10
        case .superia400:
            halationIntensity = 0.26; emulsionBloom = 0.22; filmGrain = 0.24; printPaperContrast = 1.06; printWarmthSplit = -0.06
        case .velvia50:
            halationIntensity = 0.10; emulsionBloom = 0.12; filmGrain = 0.08; printPaperContrast = 1.18; printWarmthSplit = 0.04
        case .provia100F:
            halationIntensity = 0.12; emulsionBloom = 0.14; filmGrain = 0.10; printPaperContrast = 1.06; printWarmthSplit = 0.0
        case .kodachrome64:
            halationIntensity = 0.18; emulsionBloom = 0.16; filmGrain = 0.20; printPaperContrast = 1.12; printWarmthSplit = 0.22
        case .ilfordHP5:
            halationIntensity = 0.32; emulsionBloom = 0.28; filmGrain = 0.38; printPaperContrast = 1.10; printWarmthSplit = 0.0
        }
    }

    mutating func reset() {
        let keepSystem = activeColorSystem
        let keepLUT = lutPreset
        let keepBake = bakeGradeIntoRecording
        let keepStock = filmStock
        self = ColorGradeParameters()
        self.activeColorSystem = keepSystem
        self.lutPreset = keepLUT
        self.bakeGradeIntoRecording = keepBake
        if keepSystem == .physicalFilmLab {
            self.applyStockDefaults(keepStock)
        }
    }
}

// MARK: - 3. 3D .cube LUT Parser & Exporter (Scalar Float Math + Tetrahedral Compatible)

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

    static func exportCubeFile(grade: ColorGradeParameters) throws -> URL {
        let size = 33
        let nameTag = grade.activeColorSystem == .physicalFilmLab ? grade.filmStock.shortName : grade.lutPreset.title
        var lines: [String] = [
            "# Generated by CineLog 13 Pro Max v5.0",
            "TITLE \"CineLog13Pro_\(nameTag)\"",
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
                    let out = evaluateGradeCPU(appleLogRGB: SIMD3<Float>(rIn, gIn, bIn), u: grade)
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

    private static func acesToneMapChannel(_ x: Float) -> Float {
        let num: Float = x * (2.51 * x + 0.03)
        let den: Float = x * (2.43 * x + 0.59) + 0.14
        let mapped: Float = clamp01(num / max(0.0001, den))
        return pow(mapped, 1.0 / 2.2)
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

    private static func gradeSingleLogChannel(_ val: Float, lift: Float, gamma: Float, gain: Float, pivot: Float, contrast: Float) -> Float {
        let lifted: Float = val + lift * (1.0 - val)
        let gained: Float = clamp01(lifted * gain)
        let gPow: Float = max(0.2, 1.0 - gamma * 0.65)
        let curved: Float = pow(gained, gPow)
        let contrasted: Float = (curved - pivot) * contrast + pivot
        return clamp01(contrasted)
    }

    private static func evaluateGradeCPU(appleLogRGB: SIMD3<Float>, u: ColorGradeParameters) -> SIMD3<Float> {
        let tempShift: Float = u.temperatureOffset * 0.06
        let tintShift: Float = u.tintOffset * 0.03

        let r0: Float = appleLogRGB.x + tempShift + tintShift
        let g0: Float = appleLogRGB.y - u.tintOffset * 0.05
        let b0: Float = appleLogRGB.z - tempShift + tintShift

        let rLog: Float = gradeSingleLogChannel(r0, lift: u.liftR, gamma: u.gammaR, gain: u.gainR, pivot: u.pivot, contrast: u.contrast)
        let gLog: Float = gradeSingleLogChannel(g0, lift: u.liftG, gamma: u.gammaG, gain: u.gainG, pivot: u.pivot, contrast: u.contrast)
        let bLog: Float = gradeSingleLogChannel(b0, lift: u.liftB, gamma: u.gammaB, gain: u.gainB, pivot: u.pivot, contrast: u.contrast)

        let evGain: Float = exp2(u.exposureEV)
        let linR2020: Float = max(0.0, logToLinChannel(rLog)) * evGain
        let linG2020: Float = max(0.0, logToLinChannel(gLog)) * evGain
        let linB2020: Float = max(0.0, logToLinChannel(bLog)) * evGain

        let linR709: Float = max(0.0,  1.6604910 * linR2020 - 0.5876411 * linG2020 - 0.0728499 * linB2020) * 1.45
        let linG709: Float = max(0.0, -0.1245505 * linR2020 + 1.1328999 * linG2020 - 0.0083494 * linB2020) * 1.45
        let linB709: Float = max(0.0, -0.0181508 * linR2020 - 0.1005789 * linG2020 + 1.1187297 * linB2020) * 1.45

        let recR: Float = acesToneMapChannel(linR709)
        let recG: Float = acesToneMapChannel(linG709)
        let recB: Float = acesToneMapChannel(linB709)

        let luma: Float = 0.2126 * recR + 0.7152 * recG + 0.0722 * recB
        let outR: Float = clamp01(luma + (recR - luma) * u.saturation)
        let outG: Float = clamp01(luma + (recG - luma) * u.saturation)
        let outB: Float = clamp01(luma + (recB - luma) * u.saturation)

        return SIMD3<Float>(outR, outG, outB)
    }
}

// MARK: - 4. Core 10-Bit Apple Log + Dual Color Engine + Non-Blocking GPU Pipeline

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

    // Multi-Stage Professional Stabilization & Gyro Horizon Leveler
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

    // Dual Color Engine (LUT System + Physical Film Lab) & Monitoring Controls
    @Published var grade = ColorGradeParameters()
    @Published var monitorDiagnostic: MonitorDiagnosticMode = .off
    @Published var aspectGuide: AspectGuideOption = .none
    @Published var showGridOverlay: Bool = true
    @Published var customLUTName: String = "未加载 .cube"

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

    // Semaphore limiting in-flight GPU frames to 2 (Eliminates blocking waitUntilCompleted() stutter!)
    private let gpuFrameSemaphore = DispatchSemaphore(value: 2)
    private var outputBufferPool: CVPixelBufferPool?
    private var previewBufferPool: CVPixelBufferPool?
    private var currentPoolDimensions = CMVideoDimensions(width: 0, height: 0)

    private let previewLock = NSLock()
    private var _latestPreviewTexture: MTLTexture?
    private var _latestRecordTexture: MTLTexture?
    private var pendingPhotoCaptureRequest: Bool = false
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
        self.ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.displayP3) as Any])
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
            motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
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
            self.grade.selectLUTPreset(.customCubeLUT)
            self.statusMessage = "四面体插值已加载 LUT: \(name)"
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
        let bitStr = isTenBit ? "10-Bit BT.2020" : "P3 + DeBand 16-Bit"

        DispatchQueue.main.async {
            self.minISO = minISOVal
            self.maxISO = maxISOVal
            self.currentISO = currentISOVal
            self.activeResolutionBadge = "\(dims.width)×\(dims.height) (\(bitStr))"
            self.activePipelineBadge = "\(self.selectedLens.focalLabel)"
            self.statusMessage = "高速双引擎管线已就绪"
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

    // MARK: - High-Bit-Depth Still Photo Capture

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

        videoProcessingQueue.asyncAfter(deadline: .now() + 0.05) { [weak self] in
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

            let colorSpace = CGColorSpace(name: CGColorSpace.displayP3) ?? CGColorSpaceCreateDeviceRGB()
            if var ciImg = CIImage(mtlTexture: targetTex, options: [.colorSpace: colorSpace]) {
                ciImg = ciImg.transformed(by: CGAffineTransform(scaleX: 1.0, y: -1.0).translatedBy(x: 0, y: -ciImg.extent.height))
                if let cgImg = ciCtx.createCGImage(ciImg, from: ciImg.extent, format: .RGBA16, colorSpace: colorSpace) {
                    let uiImage = UIImage(cgImage: cgImg)
                    self.savePhotoToLibrary(image: uiImage, label: self.grade.activeSummaryBadge)
                }
            }

            if self.dualSaveRawAndFilmPhoto, let rawTex = rawLogTex,
               var rawCI = CIImage(mtlTexture: rawTex, options: [.colorSpace: colorSpace]) {
                rawCI = rawCI.transformed(by: CGAffineTransform(scaleX: 1.0, y: -1.0).translatedBy(x: 0, y: -rawCI.extent.height))
                if let rawCG = ciCtx.createCGImage(rawCI, from: rawCI.extent, format: .RGBA16, colorSpace: colorSpace) {
                    let rawUI = UIImage(cgImage: rawCG)
                    self.savePhotoToLibrary(image: rawUI, label: "AppleLog_Raw")
                }
            }
        }
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

    // MARK: - Rock-Solid 10-Bit ProRes / HEVC Recording Control

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

            let rawW = Int(self.currentPoolDimensions.width > 0 ? self.currentPoolDimensions.width : 3840)
            let rawH = Int(self.currentPoolDimensions.height > 0 ? self.currentPoolDimensions.height : 2160)
            let width = (rawW / 2) * 2
            let height = (rawH / 2) * 2

            let takeStr = String(format: "T%02d", self.clipTakeNumber)
            let fileName = "\(self.projectReelName)_\(takeStr)_\(width)x\(height)_\(Int(Date().timeIntervalSince1970)).mov"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try? FileManager.default.removeItem(at: outputURL)

            do {
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

                let isRawLogRec = !self.grade.bakeGradeIntoRecording || (self.grade.activeColorSystem == .lutSystem && self.grade.lutPreset == .rawAppleLog)
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

                if !writer.canApply(outputSettings: videoSettings, forMediaType: .video) || (height > 2160 && chosenCodec != .hevc10Bit) {
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
                vInput.transform = .identity

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

                let modeDesc = self.grade.bakeGradeIntoRecording ? "直出 (\(self.grade.activeSummaryBadge))" : "纯净 Apple Log"
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

    private func ensurePixelBufferPools(width: Int32, height: Int32) {
        let evenW = (width / 2) * 2
        let evenH = (height / 2) * 2
        if currentPoolDimensions.width == evenW && currentPoolDimensions.height == evenH &&
           outputBufferPool != nil && previewBufferPool != nil {
            return
        }
        currentPoolDimensions = CMVideoDimensions(width: evenW, height: evenH)

        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 6]
        let bufferAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
            kCVPixelBufferWidthKey as String: evenW,
            kCVPixelBufferHeightKey as String: evenH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &outputBufferPool)
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &previewBufferPool)
    }
}

// MARK: - 5. Non-Blocking Pipelined AVCaptureVideoDataOutput Delegate (Zero Stutter!)

extension AppleLogCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

    struct ColorGradeUniforms {
        var isNativeAppleLog: Int32
        var isFullRangeYCbCr: Int32
        var isSDRSensorInput: Int32
        var rotationMode: Int32
        var mirrorHorizontal: Int32
        var writeRecordBuffer: Int32

        var activeColorSystem: Int32
        var lutPresetID: Int32
        var lutInputSpace: Int32
        var customLUTSize: Int32
        var bakeGradeIntoRecording: Int32
        var monitorDiagnosticMode: Int32
        var hasCustom3DLUT: Int32

        var focusPeakingEnabled: Int32
        var focusPeakingColorIndex: Int32
        var focusPeakingThreshold: Float

        var exposureEV: Float
        var contrast: Float
        var pivot: Float
        var highlightRollOff: Float
        var shadowLift: Float
        var saturation: Float
        var vibrance: Float
        var temperatureOffset: Float
        var tintOffset: Float
        var lutIntensity: Float

        var filmStockID: Int32
        var pushPullStops: Float
        var halationIntensity: Float
        var halationRadius: Float
        var emulsionBloom: Float
        var lensVignette: Float
        var chromaticAberration: Float
        var printPaperContrast: Float
        var printWarmthSplit: Float
        var filmGrain: Float
        var grainSize: Float
        var debandingStrength: Float
        var frameTime: Float

        var liftRGB: SIMD4<Float>
        var gammaRGB: SIMD4<Float>
        var gainRGB: SIMD4<Float>
    }

    func makeShaderUniforms(writeRecord: Bool) -> ColorGradeUniforms {
        let g = grade
        return ColorGradeUniforms(
            isNativeAppleLog: hardwareSupportsNativeAppleLog ? 1 : 0,
            isFullRangeYCbCr: activeFormatIsFullRange ? 1 : 0,
            isSDRSensorInput: activeFormatIsSDRFallback ? 1 : 0,
            rotationMode: Int32(orientationMode.rawValue),
            mirrorHorizontal: isMirroredHorizontal ? 1 : 0,
            writeRecordBuffer: writeRecord ? 1 : 0,
            activeColorSystem: Int32(g.activeColorSystem.rawValue),
            lutPresetID: Int32(g.lutPreset.rawValue),
            lutInputSpace: Int32(g.lutInputSpace.rawValue),
            customLUTSize: Int32(custom3DLUTSize),
            bakeGradeIntoRecording: g.bakeGradeIntoRecording ? 1 : 0,
            monitorDiagnosticMode: Int32(monitorDiagnostic.rawValue),
            hasCustom3DLUT: custom3DLUTTexture != nil ? 1 : 0,
            focusPeakingEnabled: focusPeakingEnabled ? 1 : 0,
            focusPeakingColorIndex: Int32(focusPeakingColor.rawValue),
            focusPeakingThreshold: focusPeakingThreshold,
            exposureEV: g.exposureEV,
            contrast: g.contrast,
            pivot: g.pivot,
            highlightRollOff: g.highlightRollOff,
            shadowLift: g.shadowLift,
            saturation: g.saturation,
            vibrance: g.vibrance,
            temperatureOffset: g.temperatureOffset,
            tintOffset: g.tintOffset,
            lutIntensity: g.lutIntensity,
            filmStockID: Int32(g.filmStock.rawValue),
            pushPullStops: g.pushPullStops,
            halationIntensity: g.halationIntensity,
            halationRadius: g.halationRadius,
            emulsionBloom: g.emulsionBloom,
            lensVignette: g.lensVignette,
            chromaticAberration: g.chromaticAberration,
            printPaperContrast: g.printPaperContrast,
            printWarmthSplit: g.printWarmthSplit,
            filmGrain: g.filmGrain,
            grainSize: g.grainSize,
            debandingStrength: g.debandingStrength,
            frameTime: Float(CACurrentMediaTime() - startTimestamp),
            liftRGB: SIMD4<Float>(g.liftR, g.liftG, g.liftB, 0.0),
            gammaRGB: SIMD4<Float>(g.gammaR, g.gammaG, g.gammaB, 0.0),
            gainRGB: SIMD4<Float>(g.gainR, g.gainG, g.gainB, 1.0)
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

        // Non-blocking check: if 2 GPU frames are already in flight, drop frame immediately to prevent stutter!
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

        let rawOutW = orientationMode.isPortraitSwapped ? sensorH : sensorW
        let rawOutH = orientationMode.isPortraitSwapped ? sensorW : sensorH
        let outW = (rawOutW / 2) * 2
        let outH = (rawOutH / 2) * 2
        ensurePixelBufferPools(width: Int32(outW), height: Int32(outH))

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
              let outPool = outputBufferPool,
              let prevPool = previewBufferPool else {
            gpuFrameSemaphore.signal()
            return
        }

        recordingStateLock.lock()
        let recActive = (recordingState == .recording)
        recordingStateLock.unlock()

        previewLock.lock()
        let needRecordTex = recActive || pendingPhotoCaptureRequest
        previewLock.unlock()

        var previewPixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, prevPool, &previewPixelBuffer)
        guard let prevOutPB = previewPixelBuffer else {
            gpuFrameSemaphore.signal()
            return
        }

        // Only allocate & bind logOutPixelBuffer when actively recording or snapping a still photo!
        var logOutPixelBuffer: CVPixelBuffer?
        if needRecordTex {
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outPool, &logOutPixelBuffer)
        }
        let logOutPB = logOutPixelBuffer ?? prevOutPB

        var cvLogOutTex: CVMetalTexture?
        var cvPrevOutTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, prevOutPB, nil, .rgba16Float, outW, outH, 0, &cvPrevOutTex
        )
        if needRecordTex {
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache, logOutPB, nil, .rgba16Float, outW, outH, 0, &cvLogOutTex
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
        var uniforms = makeShaderUniforms(writeRecord: needRecordTex)

        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(lumaTex, index: 0)
        encoder.setTexture(chromaTex, index: 1)
        encoder.setTexture(logOutTex, index: 2)
        encoder.setTexture(prevOutTex, index: 3)
        encoder.setTexture(custom3DLUTTexture ?? fallbackLUT, index: 4)
        encoder.setBytes(&uniforms, length: MemoryLayout<ColorGradeUniforms>.stride, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (outW + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (outH + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        // Asynchronous Non-Blocking Completion Handler (Eliminates CPU-GPU stall!)
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            self.previewLock.lock()
            self._latestPreviewTexture = prevOutTex
            if needRecordTex {
                self._latestRecordTexture = logOutTex
            }
            self.previewLock.unlock()

            if recActive {
                self.videoProcessingQueue.async {
                    self.appendRecordedPixelBuffer(logOutPB, timestamp: timestamp)
                }
            }
            self.gpuFrameSemaphore.signal()
        }
        commandBuffer.commit()

        frameCounter += 1
        if frameCounter % 12 == 0 {
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

        let step = max(1, sampleCount / 128)
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
            for y in stride(from: 0, to: height, by: 32) {
                let rowOffset = y * strideU16
                for x in stride(from: 0, to: width, by: 32) {
                    let normalized = Float(uint16Ptr[rowOffset + x]) / 65535.0
                    let binIdx = min(63, max(0, Int(normalized * 63.0)))
                    bins[binIdx] += 1
                    totalSamples += 1
                }
            }
        } else {
            let uint8Ptr = baseAddr.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: 0, to: height, by: 32) {
                let rowOffset = y * bytesPerRow
                for x in stride(from: 0, to: width, by: 32) {
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
                        enc.setBytes(&uniforms, length: MemoryLayout<ColorGradeUniforms>.stride, index: 0)

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

// MARK: - 6. EDR 10-Bit Metal Viewfinder

struct MetalLogViewfinder: UIViewRepresentable {
    @ObservedObject var engine: AppleLogCameraEngine

    func makeCoordinator() -> Coordinator {
        Coordinator(engine: engine)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: engine.metalDevice)
        mtkView.preferredFramesPerSecond = 60
        mtkView.framebufferOnly = true
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.colorPixelFormat = .bgr10a2Unorm
        mtkView.clearColor = MTLClearColorMake(0, 0, 0, 1)

        if let metalLayer = mtkView.layer as? CAMetalLayer {
            metalLayer.wantsExtendedDynamicRangeContent = true
            metalLayer.colorspace = CGColorSpace(name: CGColorSpace.itur_2020)
        }
        mtkView.delegate = context.coordinator
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    final class Coordinator: NSObject, MTKViewDelegate {
        let engine: AppleLogCameraEngine
        private let commandQueue: MTLCommandQueue?
        private var renderPipelineState: MTLRenderPipelineState?

        struct ViewfinderScaleUniforms {
            var scale: SIMD2<Float>
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
            desc.colorAttachments[0].pixelFormat = .bgr10a2Unorm
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
            let texW  = Float(max(1, previewTex.width))
            let texH  = Float(max(1, previewTex.height))

            let viewAspect = viewW / viewH
            let texAspect  = texW / texH

            var scale = SIMD2<Float>(1.0, 1.0)
            if viewAspect > texAspect {
                scale.x = texAspect / viewAspect
            } else {
                scale.y = viewAspect / texAspect
            }

            var vfUniforms = ViewfinderScaleUniforms(scale: scale)

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

// MARK: - 7. Post-Production Studio Sheet (Supports Both LUT System & Physical Film Lab)

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
    @State private var exportStatus: String = "可从相册选择视频，使用「LUT 调色系统」或「物理胶片暗房」渲染成片"
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
                        if let url = try? CubeLUTManager.exportCubeFile(grade: engine.grade) {
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

                Picker("色彩系统", selection: $engine.grade.activeColorSystem) {
                    ForEach(ActiveColorSystemMode.allCases) { sys in
                        Text(sys.title).tag(sys)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)

                ScrollView {
                    if engine.grade.activeColorSystem == .lutSystem {
                        LUTSystemDrawerView(engine: engine, showCubeImporter: .constant(false))
                            .padding(.horizontal, 16)
                    } else {
                        PhysicalFilmLabDrawerView(engine: engine)
                            .padding(.horizontal, 16)
                    }
                }
            }
            .background(Color(white: 0.08).ignoresSafeArea())
            .navigationTitle("后期实验室 (LUT & 物理暗房)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("重置参数") { engine.grade.reset() }
                        .foregroundColor(.orange)
                }
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
                            exportStatus = "已载入视频，选择下方 LUT 或物理胶片后点击「渲染导出 10-Bit」"
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

// MARK: - 8A. Independent System 1: LUT Color Grading System Drawer (10 Built-in Cinema LUTs + .cube Import)

struct LUTSystemDrawerView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @Binding var showCubeImporter: Bool
    @State private var lutSubTab: Int = 0 // 0 = 内置电影 LUT 与一级调色, 1 = LGG 三路色轮

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("系统 A：Apple Log 专业 LUT 调色台（10 款内置 + .cube 四面体插值）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                Spacer()
                Button("重置") { engine.grade.reset() }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
            }

            // 10 Built-in Cinema & Print LUT Presets Carousel
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(BuiltInLUTPresetOption.allCases) { preset in
                        Button {
                            engine.grade.selectLUTPreset(preset)
                        } label: {
                            Text(preset.title)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(engine.grade.lutPreset == preset && engine.grade.activeColorSystem == .lutSystem ? Color.yellow : Color.white.opacity(0.12))
                                .foregroundColor(engine.grade.lutPreset == preset && engine.grade.activeColorSystem == .lutSystem ? .black : .white)
                                .cornerRadius(6)
                        }
                    }
                }
            }

            Text(engine.grade.lutPreset.subtitle)
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

                Picker("LUT Space", selection: $engine.grade.lutInputSpace) {
                    ForEach(LUTInputSpaceOption.allCases) { sp in
                        Text(sp.title).tag(sp)
                    }
                }
                .pickerStyle(.segmented)
            }

            Toggle(isOn: $engine.grade.bakeGradeIntoRecording) {
                Text(engine.grade.bakeGradeIntoRecording ? "录制视频：将 LUT 调色烧录直出" : "录制视频：保存纯净 Apple Log（LUT 仅监看/拍照生效）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(engine.grade.bakeGradeIntoRecording ? .orange : .green)
            }
            .tint(.orange)

            Picker("LUT 调色分组", selection: $lutSubTab) {
                Text("一级调色 & 浓度").tag(0)
                Text("三路色轮 (Lift/Gamma/Gain)").tag(1)
            }
            .pickerStyle(.segmented)

            if lutSubTab == 0 {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "LUT 混合浓度", value: $engine.grade.lutIntensity, range: 0.0...1.0, format: "%.2f", tint: .yellow)
                    GradeSliderRow(label: "曝光补偿 EV", value: $engine.grade.exposureEV, range: -2.5...2.5, format: "%+.1f EV", tint: .yellow)
                    GradeSliderRow(label: "对数对比度", value: $engine.grade.contrast, range: 0.5...1.6, format: "%.2f", tint: .white)
                    GradeSliderRow(label: "高光柔和滚降", value: $engine.grade.highlightRollOff, range: 0.0...1.0, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "全局饱和度", value: $engine.grade.saturation, range: 0.0...2.0, format: "%.2f", tint: .pink)
                    GradeSliderRow(label: "去断层平滑", value: $engine.grade.debandingStrength, range: 0.0...1.0, format: "%.2f", tint: .cyan)
                }
            } else {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "暗部 Lift R", value: $engine.grade.liftR, range: -0.2...0.2, format: "%+.2f", tint: .red)
                    GradeSliderRow(label: "暗部 Lift G", value: $engine.grade.liftG, range: -0.2...0.2, format: "%+.2f", tint: .green)
                    GradeSliderRow(label: "暗部 Lift B", value: $engine.grade.liftB, range: -0.2...0.2, format: "%+.2f", tint: .blue)
                    GradeSliderRow(label: "中灰 Gamma B", value: $engine.grade.gammaB, range: -0.3...0.3, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "高光 Gain R", value: $engine.grade.gainR, range: 0.7...1.3, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "高光 Gain B", value: $engine.grade.gainB, range: 0.7...1.3, format: "%.2f", tint: .blue)
                }
            }
        }
    }
}

// MARK: - 8B. Independent System 2: Direct-Photon Physical Film Emulsion & Darkroom Lab Drawer (12 Classic Stocks)

struct PhysicalFilmLabDrawerView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @State private var filmSubTab: Int = 0 // 0 = 胶片乳剂与暗房显影, 1 = 光学镜头与银盐颗粒

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("系统 B：直接光子物理胶片暗房（绕过 Log • 10进制 H&D 感光密度方程）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.orange)
                Spacer()
                Button("重置暗房") {
                    engine.grade.applyStockDefaults(engine.grade.filmStock)
                }
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)
            }

            // 12 Classic Physical Film Stocks Carousel
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(PhysicalFilmStockOption.allCases) { stock in
                        Button {
                            engine.grade.applyStockDefaults(stock)
                        } label: {
                            Text(stock.shortName)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(engine.grade.filmStock == stock && engine.grade.activeColorSystem == .physicalFilmLab ? Color.orange : Color.white.opacity(0.12))
                                .foregroundColor(engine.grade.filmStock == stock && engine.grade.activeColorSystem == .physicalFilmLab ? .black : .white)
                                .cornerRadius(6)
                        }
                    }
                }
            }

            Text(engine.grade.filmStock.chemistryBadge)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundColor(.yellow)

            Toggle(isOn: $engine.grade.bakeGradeIntoRecording) {
                Text(engine.grade.bakeGradeIntoRecording ? "录制视频：将物理胶片暗房直接冲洗进视频" : "录制视频：保存纯净 Apple Log（物理胶片仅监看/拍照生效）")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(engine.grade.bakeGradeIntoRecording ? .orange : .green)
            }
            .tint(.orange)

            Picker("暗房控制分组", selection: $filmSubTab) {
                Text("显影迫冲 & 相纸印相").tag(0)
                Text("光学光晕 / 暗角 / 银盐颗粒").tag(1)
            }
            .pickerStyle(.segmented)

            if filmSubTab == 0 {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "迫冲 Push/Pull", value: $engine.grade.pushPullStops, range: -1.5...2.0, format: "%+.1f 档", tint: .orange)
                    GradeSliderRow(label: "相纸反差 Grade", value: $engine.grade.printPaperContrast, range: 0.70...1.45, format: "%.2f", tint: .white)
                    GradeSliderRow(label: "暗房冷暖染印", value: $engine.grade.printWarmthSplit, range: -1.0...1.0, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "曝光补偿 EV", value: $engine.grade.exposureEV, range: -2.5...2.5, format: "%+.1f EV", tint: .yellow)
                    GradeSliderRow(label: "暗部密度提亮", value: $engine.grade.shadowLift, range: -0.25...0.25, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "成色剂饱和度", value: $engine.grade.saturation, range: 0.0...1.8, format: "%.2f", tint: .pink)
                }
            } else {
                VStack(spacing: 5) {
                    GradeSliderRow(label: "片基红光晕", value: $engine.grade.halationIntensity, range: 0.0...1.0, format: "%.2f", tint: .red)
                    GradeSliderRow(label: "光晕扩散半径", value: $engine.grade.halationRadius, range: 1.0...8.0, format: "%.1f px", tint: .red)
                    GradeSliderRow(label: "乳剂柔光 Bloom", value: $engine.grade.emulsionBloom, range: 0.0...1.0, format: "%.2f", tint: .yellow)
                    GradeSliderRow(label: "银盐颗粒密度", value: $engine.grade.filmGrain, range: 0.0...1.0, format: "%.2f", tint: .gray)
                    GradeSliderRow(label: "银盐画幅尺寸", value: $engine.grade.grainSize, range: 0.5...1.8, format: "%.2fx", tint: .gray)
                    GradeSliderRow(label: "光学暗角 cos⁴θ", value: $engine.grade.lensVignette, range: 0.0...1.0, format: "%.2f", tint: .purple)
                    GradeSliderRow(label: "径向色散 CA", value: $engine.grade.chromaticAberration, range: 0.0...1.0, format: "%.2f", tint: .pink)
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

// MARK: - 9. Blackmagic-Style Overlays & Bottom 6-Tab Dock

enum ActiveControlTab: Int, CaseIterable, Identifiable {
    case none        = 0
    case format      = 1 // 画幅/防抖
    case focus       = 2 // 对焦/跟焦
    case exposure    = 3 // 曝光/色温
    case lutSystem   = 4 // LUT 调色 (独立系统 A)
    case physicalLab = 5 // 物理胶片 (独立系统 B)
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

                    // Row 3: Single-Line Telemetry Banner
                    HStack {
                        Text("\(engine.activeResolutionBadge) • \(engine.statusMessage)")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundColor(.white.opacity(0.88))
                            .lineLimit(1)
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
                            .background(Color.white.opacity(0.20))
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
                                        if let url = try? CubeLUTManager.exportCubeFile(grade: engine.grade) {
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

                    // Bottom Control Dock: 6 Dedicated Tabs (Separating LUT System & Physical Film Lab!)
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
                                    badgeDot: engine.grade.activeColorSystem == .lutSystem ? .yellow : nil
                                ) {
                                    engine.grade.activeColorSystem = .lutSystem
                                    toggleTab(.lutSystem)
                                }
                                DrawerTabButton(
                                    title: "物理胶片",
                                    icon: "film",
                                    isActive: activeTab == .physicalLab,
                                    badgeDot: engine.grade.activeColorSystem == .physicalFilmLab ? .orange : nil
                                ) {
                                    engine.grade.activeColorSystem = .physicalFilmLab
                                    toggleTab(.physicalLab)
                                }
                                DrawerTabButton(title: "辅助/场记", icon: "waveform.path.ecg", isActive: activeTab == .monitor) {
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
                                Text(engine.grade.activeSummaryBadge)
                                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                    .foregroundColor(engine.grade.activeColorSystem == .physicalFilmLab ? .orange : .yellow)
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
                                Text(engine.captureMode == .photo ? "模式: 16-Bit 胶片拍照" : (engine.grade.bakeGradeIntoRecording ? "REC: 烧录直出" : "REC: 纯净 APPLE LOG"))
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(engine.captureMode == .photo ? .yellow : (engine.grade.bakeGradeIntoRecording ? .orange : .green))
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