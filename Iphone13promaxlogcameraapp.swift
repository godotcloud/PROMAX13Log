//
//  iPhone13ProMaxLogCameraApp.swift
//  CineLog13Pro - Professional 10-Bit Apple Log Cinema & Still Camera for iPhone 13 Pro Max
//
//  Target: iOS 16.0+ (Tailored for iPhone 13 Pro Max A15 Bionic & compatible with iPhone 15 Pro+)
//  Frameworks: UIKit, SwiftUI, AVFoundation, Metal, MetalKit, CoreVideo, CoreMedia,
//              VideoToolbox, Photos, CoreImage, CoreMotion, UniformTypeIdentifiers
//
//  Key Updates in v10.0 (Deep Memory & Storage Purge):
//  1. Adaptive Landscape & Portrait Cinema HUD (Fixes Landscape UI Overflow in Screenshot):
//     - Automatically detects landscape vs portrait viewport geometry (geo.size.width > geo.size.height).
//     - In Landscape: merges the top telemetry/VU/histogram into a single 36pt bar, merges the bottom
//       lens/drawer-tabs/mode/shutter controls into a single locked 58pt bottom bar, arranges drawer
//       sliders in a 2-column grid, and caps drawer height with a ScrollView + "✕ 收起" button so zero
//       controls are ever pushed off-screen.
//  2. Persistent Resident 3D .cube LUT Library (驻留 LUT — Survives App Restarts):
//     - Imported .cube LUT files are permanently saved into Documents/ResidentLUTs/ and indexed in UserDefaults.
//     - Automatically restores and activates the last-used resident .cube LUT (and input color space)
//       on app launch without re-importing.
//     - Supports saving multiple resident .cube LUTs with 1-tap switching and deletion directly in the LUT drawer.
//  3. Full 4032x3024 Open Gate Video & Photo Saving + Accurate ACES/BT.709 Built-in LUTs + Zero Temp Bloat.
//

import UIKit
import SwiftUI
import AVFoundation
import Metal
import MetalKit
import CoreVideo
import CoreMedia
import VideoToolbox
import Photos
import CoreImage
import CoreMotion
import UniformTypeIdentifiers
import QuartzCore
import Combine

// MARK: - 1. Embedded Metal Compute & Render Shaders

enum MetalShaderLibrarySource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct PipelineUniforms {
        int isNativeAppleLog;
        int isOpenGateMode;          // 1 ONLY when Open Gate 4:3 (4032x3024) is active; 0 for standard 10-bit 4K/1080p
        int isFullRangeYCbCr;
        int isTenBitSensorInput;

        int isSDRSensorInput;
        int writeRecordBuffer;
        int bakeLUTIntoRecording;
        int monitorDiagnosticMode;   // 0 = Normal, 1 = False Color IRE, 2 = Zebra 94% IRE

        int focusPeakingEnabled;
        int focusPeakingColorIndex;  // 0 = Green, 1 = Red, 2 = Yellow, 3 = Blue
        float focusPeakingThreshold;
        int lutPresetID;             // 0 = Raw Apple Log, 1..8 = Built-in Cinema LUTs, 9 = Resident/Imported .cube 3D LUT

        int lutInputSpace;           // 0 = Apple Log (BT.2020 Native iPhone 15), 1 = Apple Log (Rec.709 Gamut), 2 = Rec.709
        int customLUTSize;
        int hasCustom3DLUT;
        float lutIntensity;

        float exposureEV;
        float contrast;
        float highlightRollOff;
        float shadowLift;

        float saturation;
        float vibrance;
        float temperature;
        float tint;

        float debandingStrength;
        float frameTime;
        float pad0;
        float pad1;

        float4 domainMin;
        float4 domainMax;
        float4 liftRGB;
        float4 gammaRGB;
        float4 gainRGB;
    };

    // --- 1A. Original 10-Bit Video-Range YCbCr ('x420' / 'x422') to ITU-R BT.2020 RGB (For 4K 16:9 & 1080p 16:9) ---
    inline float3 ycbcr10VideoRangeToBT2020RGB(float yRaw, float2 cbcrRaw) {
        float y10  = yRaw * 1023.0;
        float2 c10 = cbcrRaw * 1023.0;

        float y  = clamp((y10 - 64.0) / (940.0 - 64.0), 0.0, 1.0);
        float cb = (c10.x - 512.0) / (960.0 - 64.0);
        float cr = (c10.y - 512.0) / (960.0 - 64.0);

        float r = y + 1.4746 * cr;
        float g = y - 0.16455312684366 * cb - 0.57135312684366 * cr;
        float b = y + 1.8814 * cb;

        return clamp(float3(r, g, b), 0.0, 1.0);
    }

    // --- 1B. Original ITU-R BT.2100 HLG Inverse OETF to Scene-Linear BT.2020 (For 4K 16:9 & 1080p 16:9) ---
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

    inline float3 hlgToLinearBT2020(float3 hlgRGB, float evBias, float rollOff) {
        float3 sceneLinear = float3(
            hlgInverseOETFChannel(hlgRGB.r),
            hlgInverseOETFChannel(hlgRGB.g),
            hlgInverseOETFChannel(hlgRGB.b)
        );
        float gain = exp2(evBias) * 2.4;
        sceneLinear *= gain;

        if (rollOff > 0.001) {
            float3 shoulder = sceneLinear / (1.0 + sceneLinear * (0.25 * rollOff));
            sceneLinear = mix(sceneLinear, shoulder, rollOff);
        }
        return sceneLinear;
    }

    // --- 1C. Open Gate 4:3 Dedicated 8-Bit Debander & SDR Inverse OETF (Active ONLY when isOpenGateMode == 1) ---
    inline float interleavedGradientNoise(float2 pixelCoord, float framePhase) {
        float2 p = pixelCoord + float2(framePhase * 5.588238, framePhase * 3.141592);
        return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
    }

    inline float2 stochasticOffset(float2 pixelCoord, float seed, float radiusPx, float2 invTexSize) {
        float a = interleavedGradientNoise(pixelCoord + float2(seed * 17.3, seed * 31.7), seed) * 6.2831853;
        float r = (0.25 + 0.75 * interleavedGradientNoise(pixelCoord + float2(seed * 43.1, seed * 11.9), seed + 1.0)) * radiusPx;
        return float2(cos(a), sin(a)) * r * invTexSize;
    }

    inline void sampleOpenGateDebandedYCbCr(
        texture2d<float, access::sample> lumaPlane,
        texture2d<float, access::sample> chromaPlane,
        float2 uv,
        uint2 gid,
        float2 lumaSize,
        int isTenBit,
        float debandAmount,
        float frameTime,
        thread float& outY,
        thread float2& outCbCr
    ) {
        constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);

        float yCenter = lumaPlane.sample(linearSampler, uv).r;
        float2 cCenter = chromaPlane.sample(linearSampler, uv).rg;

        float effectiveDeband = (isTenBit == 0) ? max(0.75, debandAmount) : debandAmount;
        if (effectiveDeband <= 0.01) {
            outY = yCenter;
            outCbCr = cCenter;
            return;
        }

        float2 invSize = 1.0 / max(lumaSize, float2(1.0, 1.0));
        float phase = floor(frameTime * 24.0);

        float rInner = 32.0 * effectiveDeband;
        float rOuter = 96.0 * effectiveDeband;

        float2 d1 = stochasticOffset(float2(gid), 1.0 + phase * 0.13, rInner, invSize);
        float2 d2 = stochasticOffset(float2(gid), 2.0 + phase * 0.29, rOuter, invSize);

        float y1 = lumaPlane.sample(linearSampler, clamp(uv + d1, 0.0, 1.0)).r;
        float y2 = lumaPlane.sample(linearSampler, clamp(uv - d1, 0.0, 1.0)).r;
        float y3 = lumaPlane.sample(linearSampler, clamp(uv + d2, 0.0, 1.0)).r;
        float y4 = lumaPlane.sample(linearSampler, clamp(uv - d2, 0.0, 1.0)).r;

        float2 c1 = chromaPlane.sample(linearSampler, clamp(uv + d1, 0.0, 1.0)).rg;
        float2 c2 = chromaPlane.sample(linearSampler, clamp(uv - d1, 0.0, 1.0)).rg;
        float2 c3 = chromaPlane.sample(linearSampler, clamp(uv + d2, 0.0, 1.0)).rg;
        float2 c4 = chromaPlane.sample(linearSampler, clamp(uv - d2, 0.0, 1.0)).rg;

        float yThresh = 0.024 * effectiveDeband;
        float cThresh = 0.020 * effectiveDeband;

        float yAvg = (yCenter + y1 + y2 + y3 + y4) * 0.2;
        float yMaxDiff = max(max(abs(y1 - yCenter), abs(y2 - yCenter)), max(abs(y3 - yCenter), abs(y4 - yCenter)));
        float yWeight = smoothstep(yThresh, yThresh * 0.15, yMaxDiff);
        float ySmooth = mix(yCenter, yAvg, yWeight);

        float2 cAvg = (cCenter + c1 + c2 + c3 + c4) * 0.2;
        float2 cMaxDiff = max(max(abs(c1 - cCenter), abs(c2 - cCenter)), max(abs(c3 - cCenter), abs(c4 - cCenter)));
        float2 cWeight = smoothstep(float2(cThresh), float2(cThresh * 0.15), cMaxDiff);
        float2 cSmooth = mix(cCenter, cAvg, cWeight);

        float ign1 = interleavedGradientNoise(float2(gid), phase);
        float ign2 = interleavedGradientNoise(float2(gid) + float2(19.3, 47.1), phase);
        float tpdf = (ign1 + ign2 - 1.0);
        float ditherAmp = 0.85 / 255.0;

        outY = clamp(ySmooth + tpdf * ditherAmp, 0.0, 1.0);
        outCbCr = clamp(cSmooth + float2(tpdf * ditherAmp * 0.45, -tpdf * ditherAmp * 0.45), 0.0, 1.0);
    }

    inline float3 decodeOpenGateYCbCrToLinearBT2020(
        float yRaw,
        float2 cbcrRaw,
        int isFullRange,
        int isSDR,
        float evBias,
        float rollOff
    ) {
        float y, cb, cr;
        if (isFullRange == 1) {
            y  = clamp(yRaw, 0.0, 1.0);
            cb = cbcrRaw.x - 0.5;
            cr = cbcrRaw.y - 0.5;
        } else {
            float y8  = yRaw * 255.0;
            float2 c8 = cbcrRaw * 255.0;
            y  = clamp((y8 - 16.0) / (235.0 - 16.0), 0.0, 1.0);
            cb = (c8.x - 128.0) / (240.0 - 16.0);
            cr = (c8.y - 128.0) / (240.0 - 16.0);
        }

        if (isSDR == 1) {
            float3 sdrRGB = clamp(float3(
                y + 1.5748 * cr,
                y - 0.187324 * cb - 0.468124 * cr,
                y + 1.8556 * cb
            ), 0.0, 1.0);

            float3 linP3;
            linP3.r = (sdrRGB.r <= 0.04045) ? (sdrRGB.r / 12.92) : pow((sdrRGB.r + 0.055) / 1.055, 2.4);
            linP3.g = (sdrRGB.g <= 0.04045) ? (sdrRGB.g / 12.92) : pow((sdrRGB.g + 0.055) / 1.055, 2.4);
            linP3.b = (sdrRGB.b <= 0.04045) ? (sdrRGB.b / 12.92) : pow((sdrRGB.b + 0.055) / 1.055, 2.4);

            float3 lin2020;
            lin2020.r =  0.753833 * linP3.r + 0.198597 * linP3.g + 0.047570 * linP3.b;
            lin2020.g =  0.045744 * linP3.r + 0.941777 * linP3.g + 0.012479 * linP3.b;
            lin2020.b = -0.001210 * linP3.r + 0.017602 * linP3.g + 0.983608 * linP3.b;
            lin2020 = max(lin2020, 0.0) * exp2(evBias);

            if (rollOff > 0.001) {
                float3 shoulder = lin2020 / (1.0 + lin2020 * (0.25 * rollOff));
                lin2020 = mix(lin2020, shoulder, rollOff);
            }
            return lin2020;
        } else {
            float3 hlgRGB = clamp(float3(
                y + 1.4746 * cr,
                y - 0.16455312684366 * cb - 0.57135312684366 * cr,
                y + 1.8814 * cb
            ), 0.0, 1.0);
            return hlgToLinearBT2020(hlgRGB, evBias, rollOff);
        }
    }

    // --- 2. Official Apple Log Profile White Paper OETF & EOTF ---
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

    inline float3 linearBT2020ToLinear709(float3 lin2020) {
        float3 lin709;
        lin709.r =  1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b;
        lin709.g = -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b;
        lin709.b = -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b;
        return max(lin709, 0.0);
    }

    // --- 3. Mathematically Accurate Official ITU-R BT.709 OETF & ACES Narkowicz Filmic Transform ---
    inline float bt709OETFChannel(float L) {
        float x = clamp(L, 0.0, 1.0);
        return (x < 0.018) ? (4.5 * x) : (1.099 * pow(x, 0.45) - 0.099);
    }

    inline float3 accurateAppleLogToRec709(float3 lin709) {
        float3 x = max(lin709, 0.0) * 0.92;
        float3 acesLin = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
        float3 acesDisplay = pow(acesLin, float3(1.0 / 2.2));

        float3 compressed709 = lin709 / (1.0 + lin709 * 0.28);
        float3 std709 = float3(
            bt709OETFChannel(compressed709.r),
            bt709OETFChannel(compressed709.g),
            bt709OETFChannel(compressed709.b)
        );

        return clamp(mix(std709, acesDisplay, 0.55), 0.0, 1.0);
    }

    // --- 4. Tetrahedral 3D .cube LUT Interpolation ---
    inline float3 sample3DLUTTetrahedral(
        texture3d<float, access::read> lutTex,
        float3 rgbIn,
        int lutSize,
        float3 domainMin,
        float3 domainMax
    ) {
        int N = max(2, lutSize);
        float3 span = max(domainMax - domainMin, float3(0.0001));
        float3 normIn = clamp((rgbIn - domainMin) / span, 0.0, 1.0);

        float3 scaled = normIn * float(N - 1);
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

    // --- 5. Calibrated Cinema Built-In LUTs & 3D .cube LUT Evaluator ---
    inline float3 evaluateColorGrade(
        float3 appleLogBT2020,
        constant PipelineUniforms& u,
        texture3d<float, access::read> customLUT
    ) {
        float3 logColor = appleLogBT2020;

        logColor.r += u.temperature * 0.045 + u.tint * 0.025;
        logColor.g -= u.tint * 0.040;
        logColor.b -= u.temperature * 0.045 - u.tint * 0.025;

        float3 lifted = logColor + u.liftRGB.xyz * (1.0 - logColor);
        float3 gained = lifted * u.gainRGB.xyz;
        float3 gammaPower = max(float3(0.3), 1.0 - u.gammaRGB.xyz * 0.55);
        logColor = pow(clamp(gained, 0.0, 1.0), gammaPower);

        if (abs(u.shadowLift) > 0.001) {
            float shadowWeight = clamp(1.0 - (logColor.g / 0.5), 0.0, 1.0);
            logColor += u.shadowLift * shadowWeight * shadowWeight * 0.22;
        }
        logColor = clamp((logColor - 0.488) * u.contrast + 0.488, 0.0, 1.0);

        if (u.lutPresetID == 0) {
            float luma = dot(logColor, float3(0.2627, 0.6780, 0.0593));
            return clamp(mix(float3(luma), logColor, u.saturation), 0.0, 1.0);
        }

        float3 lin2020 = appleLogToLinear(logColor);
        float3 lin709  = linearBT2020ToLinear709(lin2020);
        float3 rec709  = accurateAppleLogToRec709(lin709);

        float3 styled = rec709;
        float luma = dot(rec709, float3(0.2126, 0.7152, 0.0722));

        if (u.lutPresetID == 2) {
            // ARRI Alexa K1S1 709
            float3 arriLin;
            arriLin.r = dot(lin709, float3( 1.04, -0.03, -0.01));
            arriLin.g = dot(lin709, float3(-0.02,  1.03, -0.01));
            arriLin.b = dot(lin709, float3(-0.01, -0.03,  1.04));
            arriLin = max(arriLin, 0.0);
            float3 x = arriLin * 0.88;
            float3 k1s1 = (x * (2.35 * x + 0.045)) / (x * (2.28 * x + 0.68) + 0.16);
            styled = pow(clamp(k1s1, 0.0, 1.0), float3(1.0 / 2.18));
        } else if (u.lutPresetID == 3) {
            // Kodak Vision3 2383 Print (D65)
            float3 kMat;
            kMat.r = dot(rec709, float3( 1.06, -0.04, -0.02));
            kMat.g = dot(rec709, float3(-0.02,  1.03, -0.01));
            kMat.b = dot(rec709, float3(-0.03,  0.02,  1.01));
            float3 printS = smoothstep(0.025, 0.975, clamp(kMat, 0.0, 1.0));
            float3 sCurve = mix(clamp(kMat, 0.0, 1.0), printS, 0.52);
            float shMask = pow(clamp(1.0 - luma, 0.0, 1.0), 1.8);
            float midMask = 4.0 * luma * (1.0 - luma);
            sCurve += float3(-0.035, 0.004, 0.042) * shMask;
            sCurve += float3( 0.018, 0.006, -0.015) * midMask;
            styled = clamp(sCurve, 0.0, 1.0);
        } else if (u.lutPresetID == 4) {
            // Kodak Vision3 2383 Print (Warm D55)
            float3 kMat;
            kMat.r = dot(rec709, float3( 1.07, -0.04, -0.03));
            kMat.g = dot(rec709, float3(-0.01,  1.02, -0.01));
            kMat.b = dot(rec709, float3(-0.04,  0.01,  1.03));
            float3 printS = smoothstep(0.025, 0.975, clamp(kMat, 0.0, 1.0));
            float3 sCurve = mix(clamp(kMat, 0.0, 1.0), printS, 0.50);
            float shMask = pow(clamp(1.0 - luma, 0.0, 1.0), 1.7);
            float hiMask = pow(clamp(luma, 0.0, 1.0), 1.1);
            sCurve += float3(-0.025, 0.006, 0.030) * shMask;
            sCurve += float3( 0.052, 0.028, -0.045) * hiMask;
            styled = clamp(sCurve, 0.0, 1.0);
        } else if (u.lutPresetID == 5) {
            // Fujifilm Eterna 3513DI Print
            float3 fMat;
            fMat.r = dot(rec709, float3( 1.01, -0.02,  0.01));
            fMat.g = dot(rec709, float3(-0.03,  1.06, -0.03));
            fMat.b = dot(rec709, float3(-0.01,  0.04,  0.97));
            float3 printS = smoothstep(0.02, 0.985, clamp(fMat, 0.0, 1.0));
            float3 sCurve = mix(clamp(fMat, 0.0, 1.0), printS, 0.38);
            float shMask = pow(clamp(1.0 - luma, 0.0, 1.0), 1.6);
            sCurve += float3(-0.015, 0.014, 0.025) * shMask;
            styled = clamp(sCurve, 0.0, 1.0);
        } else if (u.lutPresetID == 6) {
            // Teal & Orange Master
            float3 base = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.45);
            float shMask = pow(clamp(1.0 - luma, 0.0, 1.0), 1.5);
            float hiMask = pow(clamp(luma, 0.0, 1.0), 1.0);
            float3 sh = float3(-0.058, 0.015, 0.068) * shMask;
            float3 hi = float3( 0.062, 0.022, -0.048) * hiMask;
            styled = clamp(base + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 7) {
            // Bleach Bypass Silver Retention
            float3 desat = mix(float3(luma), rec709, 0.42);
            float silverOverlay = smoothstep(0.04, 0.96, luma);
            float3 bypassed = mix(desat, desat * float3(silverOverlay * 1.35 + 0.15), 0.55);
            styled = clamp(mix(bypassed, smoothstep(0.03, 0.97, bypassed), 0.45), 0.0, 1.0);
        } else if (u.lutPresetID == 8) {
            // Cyber Nocturne
            float3 base = mix(rec709, smoothstep(0.02, 0.98, rec709), 0.40);
            float shMask = pow(clamp(1.0 - luma, 0.0, 1.0), 1.4);
            float hiMask = pow(clamp(luma, 0.0, 1.0), 1.1);
            float3 sh = float3( 0.018, -0.025, 0.075) * shMask;
            float3 hi = float3(-0.025,  0.042, 0.055) * hiMask;
            styled = clamp(base + sh + hi, 0.0, 1.0);
        } else if (u.lutPresetID == 9 && u.hasCustom3DLUT == 1) {
            float3 lutSource;
            if (u.lutInputSpace == 0) {
                lutSource = logColor;
            } else if (u.lutInputSpace == 1) {
                lutSource = linearToAppleLog(lin709);
            } else {
                lutSource = rec709;
            }
            styled = sample3DLUTTetrahedral(customLUT, lutSource, u.customLUTSize, u.domainMin.xyz, u.domainMax.xyz);
        }

        float3 result = mix(rec709, styled, clamp(u.lutIntensity, 0.0, 1.0));
        float finalLuma = dot(result, float3(0.2126, 0.7152, 0.0722));
        float maxC = max(result.r, max(result.g, result.b));
        float minC = min(result.r, min(result.g, result.b));
        float vibranceFactor = 1.0 + u.vibrance * (1.0 - (maxC - minC));
        return clamp(mix(float3(finalLuma), result, max(0.0, u.saturation * vibranceFactor)), 0.0, 1.0);
    }

    // --- 6. False Color IRE Exposure Diagnostic Map ---
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

    // --- 7. Fast Sobel Edge Detection for Focus Peaking ---
    inline float computeSobelEdge(texture2d<float, access::sample> lumaPlane, float2 uv, float2 invSize) {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        float l = lumaPlane.sample(s, uv - float2(invSize.x, 0.0)).r;
        float r = lumaPlane.sample(s, uv + float2(invSize.x, 0.0)).r;
        float t = lumaPlane.sample(s, uv - float2(0.0, invSize.y)).r;
        float b = lumaPlane.sample(s, uv + float2(0.0, invSize.y)).r;
        return abs(r - l) * 2.0 + abs(b - t) * 2.0;
    }

    // --- 8. Main Camera Compute Kernel ---
    kernel void processAppleLogFrame(
        texture2d<float, access::sample> lumaPlane     [[texture(0)]],
        texture2d<float, access::sample> chromaPlane   [[texture(1)]],
        texture2d<float, access::write>  recordOutTex  [[texture(2)]],
        texture2d<float, access::write>  previewOutTex [[texture(3)]],
        texture3d<float, access::read>   customLUTTex  [[texture(4)]],
        constant PipelineUniforms&       uniforms      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = previewOutTex.get_width();
        uint outH = previewOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }

        uint lumaW = lumaPlane.get_width();
        uint lumaH = lumaPlane.get_height();
        float2 lumaSize = float2(lumaW, lumaH);
        float2 uv = (float2(gid) + 0.5) / float2(outW, outH);

        float3 appleLogPixel;

        if (uniforms.isOpenGateMode == 0) {
            // === ORIGINAL 10-BIT PIPELINE FOR 4K UHD 16:9 & 1080p 16:9 ===
            uint2 srcCoord = uint2(
                min(gid.x * lumaW / outW, lumaW - 1),
                min(gid.y * lumaH / outH, lumaH - 1)
            );
            uint2 chromaCoord = uint2(
                srcCoord.x * chromaPlane.get_width() / lumaW,
                srcCoord.y * chromaPlane.get_height() / lumaH
            );

            float  ySample    = lumaPlane.read(srcCoord).r;
            float2 cbcrSample = chromaPlane.read(chromaCoord).rg;

            float3 decodedRGB = ycbcr10VideoRangeToBT2020RGB(ySample, cbcrSample);

            if (uniforms.isNativeAppleLog == 1) {
                float3 linear = appleLogToLinear(decodedRGB) * exp2(uniforms.exposureEV);
                appleLogPixel = linearToAppleLog(linear);
            } else {
                float3 linearBT2020 = hlgToLinearBT2020(decodedRGB, uniforms.exposureEV, uniforms.highlightRollOff);
                appleLogPixel = linearToAppleLog(linearBT2020);
            }
        } else {
            // === OPEN GATE 4:3 (4032x3024) DEDICATED PIPELINE ===
            float ySample = 0.0;
            float2 cbcrSample = float2(0.5, 0.5);
            sampleOpenGateDebandedYCbCr(
                lumaPlane,
                chromaPlane,
                uv,
                gid,
                lumaSize,
                uniforms.isTenBitSensorInput,
                uniforms.debandingStrength,
                uniforms.frameTime,
                ySample,
                cbcrSample
            );

            float3 linearBT2020 = decodeOpenGateYCbCrToLinearBT2020(
                ySample,
                cbcrSample,
                uniforms.isFullRangeYCbCr,
                uniforms.isSDRSensorInput,
                uniforms.exposureEV,
                uniforms.highlightRollOff
            );
            appleLogPixel = linearToAppleLog(linearBT2020);
        }

        float3 gradedColor = evaluateColorGrade(appleLogPixel, uniforms, customLUTTex);

        if (uniforms.writeRecordBuffer == 1) {
            float3 recOut = (uniforms.bakeLUTIntoRecording == 1) ? gradedColor : appleLogPixel;
            recordOutTex.write(float4(recOut, 1.0), gid);
        }

        float3 monitorRGB = gradedColor;
        float logLuma = dot(appleLogPixel, float3(0.2627, 0.6780, 0.0593));

        if (uniforms.monitorDiagnosticMode == 1) {
            monitorRGB = falseColorMap(logLuma);
        } else if (uniforms.monitorDiagnosticMode == 2 && logLuma >= 0.92) {
            float stripe = fmod(float(gid.x + gid.y) + uniforms.frameTime * 28.0, 14.0);
            if (stripe < 7.0) {
                monitorRGB = float3(1.0, 0.15, 0.15);
            }
        }

        if (uniforms.focusPeakingEnabled == 1) {
            float edgeMag = computeSobelEdge(lumaPlane, uv, 1.0 / max(lumaSize, float2(1.0, 1.0)));
            if (edgeMag > uniforms.focusPeakingThreshold) {
                float3 peakColor = float3(0.1, 1.0, 0.25);
                if (uniforms.focusPeakingColorIndex == 1) peakColor = float3(1.0, 0.15, 0.15);
                else if (uniforms.focusPeakingColorIndex == 2) peakColor = float3(1.0, 0.92, 0.1);
                else if (uniforms.focusPeakingColorIndex == 3) peakColor = float3(0.15, 0.65, 1.0);
                monitorRGB = mix(monitorRGB, peakColor, 0.85);
            }
        }

        previewOutTex.write(float4(monitorRGB, 1.0), gid);
    }

    // --- 9. Post-Production Non-Destructive Apple Log Video / Photo Grading Kernel ---
    kernel void gradeExistingLogFrame(
        texture2d<float, access::read>  sourceLogTex [[texture(0)]],
        texture2d<float, access::write> destTex      [[texture(1)]],
        texture3d<float, access::read>  customLUTTex [[texture(2)]],
        constant PipelineUniforms&      uniforms     [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= destTex.get_width() || gid.y >= destTex.get_height()) {
            return;
        }
        float3 inLog = sourceLogTex.read(gid).rgb;
        float3 lin2020 = appleLogToLinear(inLog) * exp2(uniforms.exposureEV);
        if (uniforms.highlightRollOff > 0.001) {
            float3 sh = lin2020 / (1.0 + lin2020 * (0.25 * uniforms.highlightRollOff));
            lin2020 = mix(lin2020, sh, uniforms.highlightRollOff);
        }
        float3 adjustedLog = linearToAppleLog(lin2020);
        float3 outColor = evaluateColorGrade(adjustedLog, uniforms, customLUTTex);
        destTex.write(float4(outColor, 1.0), gid);
    }

    // --- 10. Viewfinder Render Pipeline ---
    struct ViewfinderVertexUniforms {
        float2 scale;
        int orientationMode;
        int mirrorHorizontal;
    };

    struct VertexOut {
        float4 position [[position]];
        float2 texCoord;
    };

    vertex VertexOut vertexViewfinder(
        uint vertexID [[vertex_id]],
        constant ViewfinderVertexUniforms& vUniforms [[buffer(0)]]
    ) {
        const float2 positions[4] = {
            float2(-1.0, -1.0),
            float2( 1.0, -1.0),
            float2(-1.0,  1.0),
            float2( 1.0,  1.0)
        };
        const float2 stdUVs[4] = {
            float2(0.0, 1.0),
            float2(1.0, 1.0),
            float2(0.0, 0.0),
            float2(1.0, 0.0)
        };

        float2 uv = stdUVs[vertexID];
        float2 rotatedUV = uv;
        if (vUniforms.orientationMode == 0) {
            rotatedUV = float2(uv.y, 1.0 - uv.x);
        } else if (vUniforms.orientationMode == 1) {
            rotatedUV = uv;
        } else if (vUniforms.orientationMode == 2) {
            rotatedUV = float2(1.0 - uv.x, 1.0 - uv.y);
        } else {
            rotatedUV = float2(1.0 - uv.y, uv.x);
        }

        if (vUniforms.mirrorHorizontal == 1) {
            rotatedUV.x = 1.0 - rotatedUV.x;
        }

        VertexOut out;
        out.position = float4(positions[vertexID] * vUniforms.scale, 0.0, 1.0);
        out.texCoord = rotatedUV;
        return out;
    }

    fragment float4 fragmentViewfinder(
        VertexOut in [[stage_in]],
        texture2d<float, access::sample> previewTex [[texture(0)]]
    ) {
        constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        return previewTex.sample(linearSampler, in.texCoord);
    }
    """
}

// MARK: - 2. Camera & Color Grading Configuration Models

enum CaptureModeOption: Int, CaseIterable, Identifiable {
    case video = 0
    case photo = 1

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .video: return "视频 VIDEO"
        case .photo: return "拍照 PHOTO"
        }
    }
}

enum BuiltInLUTPreset: Int, CaseIterable, Identifiable {
    case rawAppleLog     = 0
    case naturalRec709   = 1
    case arriAlexa709    = 2
    case kodak2383D65    = 3
    case kodak2383Warm   = 4
    case fuji3513DI      = 5
    case tealAndOrange   = 6
    case bleachBypass    = 7
    case cyberNocturne   = 8
    case customCubeLUT   = 9

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .rawAppleLog:   return "Raw Apple Log"
        case .naturalRec709: return "Official Rec.709"
        case .arriAlexa709:  return "ARRI Alexa K1S1"
        case .kodak2383D65:  return "Kodak 2383 D65"
        case .kodak2383Warm: return "Kodak 2383 D55"
        case .fuji3513DI:    return "Fuji 3513DI Print"
        case .tealAndOrange: return "Teal & Orange"
        case .bleachBypass:  return "Bleach Bypass"
        case .cyberNocturne: return "Cyber Nocturne"
        case .customCubeLUT: return "驻留 .cube LUT"
        }
    }
}

enum LUTInputSpaceOption: Int, CaseIterable, Identifiable {
    case appleLogBT2020 = 0
    case appleLogRec709 = 1
    case rec709Display  = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .appleLogBT2020: return "iPhone15 Log"
        case .appleLogRec709: return "Log(709适配)"
        case .rec709Display:  return "Rec.709标准"
        }
    }
}

/// Represents a permanently saved resident .cube 3D LUT file inside Documents/ResidentLUTs/
struct ResidentLUTItem: Identifiable, Equatable {
    let id: String        // fileName
    let displayName: String
    let fileName: String
    let fileURL: URL
}

struct LUTGradingParameters: Equatable {
    var lutPreset: BuiltInLUTPreset = .naturalRec709
    var lutInputSpace: LUTInputSpaceOption = .appleLogBT2020
    var lutIntensity: Float = 1.0
    var exposureEV: Float = 0.0
    var contrast: Float = 1.0
    var highlightRollOff: Float = 0.25
    var shadowLift: Float = 0.0
    var saturation: Float = 1.0
    var vibrance: Float = 0.0
    var temperature: Float = 0.0
    var tint: Float = 0.0
    var debanding: Float = 0.80

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
        self = LUTGradingParameters(lutPreset: self.lutPreset, lutInputSpace: self.lutInputSpace)
    }
}

enum SensorFormatPreset: String, CaseIterable, Identifiable {
    case uhd4K16x9    = "4K UHD 16:9 (3840×2160)"
    case fhd1080p16x9 = "1080p 16:9 (1920×1080)"
    case openGateMax  = "Open Gate 4:3 (4032×3024)"

    var id: String { rawValue }

    var shortBadge: String {
        switch self {
        case .uhd4K16x9:    return "4K 16:9"
        case .fhd1080p16x9: return "1080p 16:9"
        case .openGateMax:  return "Open Gate 4:3"
        }
    }
}

enum CaptureOrientationMode: Int, CaseIterable, Identifiable {
    case portrait           = 0
    case landscapeRight     = 1
    case landscapeLeft      = 2
    case portraitUpsideDown = 3

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .portrait:           return "竖屏 9:16 / 3:4"
        case .landscapeRight:     return "横屏 16:9 / 4:3 (右)"
        case .landscapeLeft:      return "横屏 16:9 / 4:3 (左)"
        case .portraitUpsideDown: return "竖屏倒置"
        }
    }

    var shortLabel: String {
        switch self {
        case .portrait:           return "竖屏"
        case .landscapeRight:     return "横屏(右)"
        case .landscapeLeft:      return "横屏(左)"
        case .portraitUpsideDown: return "倒置竖屏"
        }
    }

    var isPortraitSwapped: Bool {
        return self == .portrait || self == .portraitUpsideDown
    }

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

enum CinemaStabilizationMode: Int, CaseIterable, Identifiable {
    case off               = 0
    case standard          = 1
    case cinematic         = 2
    case cinematicExtended = 3

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .off:               return "OFF 关闭 (零裁切)"
        case .standard:          return "Standard 标准光学防抖"
        case .cinematic:         return "Cinematic 电影级防抖"
        case .cinematicExtended: return "Extreme 超强运动防抖"
        }
    }

    var shortTitle: String {
        switch self {
        case .off:               return "OFF"
        case .standard:          return "STD"
        case .cinematic:         return "CINE"
        case .cinematicExtended: return "EXT"
        }
    }

    var avMode: AVCaptureVideoStabilizationMode {
        switch self {
        case .off:               return .off
        case .standard:          return .standard
        case .cinematic:         return .cinematic
        case .cinematicExtended:
            if #available(iOS 15.0, *) {
                return .cinematicExtended
            } else {
                return .cinematic
            }
        }
    }
}

enum FocusControlMode: Int, CaseIterable, Identifiable {
    case continuousAF = 0
    case singleAF     = 1
    case manualFocus  = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .continuousAF: return "CAF 连续自动"
        case .singleAF:     return "SAF 单次锁定"
        case .manualFocus:  return "MF 手动对焦"
        }
    }

    var shortTitle: String {
        switch self {
        case .continuousAF: return "CAF"
        case .singleAF:     return "SAF"
        case .manualFocus:  return "MF"
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
        case .full: return "Full 全域"
        case .near: return "Near 近景特写"
        case .far:  return "Far 远景无限"
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

enum FocusPeakingColorOption: Int, CaseIterable, Identifiable {
    case green  = 0
    case red    = 1
    case yellow = 2
    case blue   = 3

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .green:  return "翠绿"
        case .red:    return "绯红"
        case .yellow: return "明黄"
        case .blue:   return "电蓝"
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

    var isProRes: Bool {
        return self != .hevc10Bit
    }

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

// MARK: - 3. 3D .cube LUT Parser, Resident LUT Persistence Manager & Exporter

struct LoadedCubeLUT {
    let texture: MTLTexture
    let size: Int
    let name: String
    let domainMin: SIMD3<Float>
    let domainMax: SIMD3<Float>
}

enum CubeLUTManager {
    private static let residentFolderName = "ResidentLUTs"

    /// Permanent directory in Documents/ResidentLUTs/ (never touched by temporary directory purges)
    static var residentLUTDirectoryURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let dir = docs.appendingPathComponent(residentFolderName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func listSavedResidentLUTs() -> [ResidentLUTItem] {
        let fm = FileManager.default
        let dir = residentLUTDirectoryURL
        guard let urls = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
            return []
        }
        return urls
            .filter { $0.pathExtension.lowercased() == "cube" }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            .map { url in
                ResidentLUTItem(
                    id: url.lastPathComponent,
                    displayName: url.deletingPathExtension().lastPathComponent,
                    fileName: url.lastPathComponent,
                    fileURL: url
                )
            }
    }

    static func saveToResidentLibrary(sourceURL: URL) throws -> URL {
        let accessed = sourceURL.startAccessingSecurityScopedResource()
        defer { if accessed { sourceURL.stopAccessingSecurityScopedResource() } }

        let rawFileName = sourceURL.lastPathComponent
        let safeFileName = rawFileName.hasSuffix(".cube") ? rawFileName : "\(rawFileName).cube"
        let destURL = residentLUTDirectoryURL.appendingPathComponent(safeFileName)

        if sourceURL.standardizedFileURL != destURL.standardizedFileURL {
            let contentData = try Data(contentsOf: sourceURL)
            try contentData.write(to: destURL, options: .atomic)
        }
        return destURL
    }

    static func loadCubeLUT(from url: URL, device: MTLDevice) throws -> LoadedCubeLUT {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        let content = try String(contentsOf: url, encoding: .utf8)
        var size3D = 0
        var size1D = 0
        var domainMin = SIMD3<Float>(0.0, 0.0, 0.0)
        var domainMax = SIMD3<Float>(1.0, 1.0, 1.0)
        var rawTriplets: [(Float, Float, Float)] = []
        rawTriplets.reserveCapacity(65 * 65 * 65)

        for rawLine in content.components(separatedBy: .newlines) {
            let cleanLine: String
            if let hashIdx = rawLine.firstIndex(of: "#") {
                cleanLine = String(rawLine[..<hashIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                cleanLine = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if cleanLine.isEmpty || cleanLine.hasPrefix("TITLE") {
                continue
            }

            let parts = cleanLine.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard !parts.isEmpty else { continue }

            let keyword = parts[0].uppercased()
            if keyword == "LUT_3D_SIZE", parts.count >= 2, let n = Int(parts[1]), n >= 2 && n <= 129 {
                size3D = n
                continue
            }
            if keyword == "LUT_1D_SIZE", parts.count >= 2, let n1 = Int(parts[1]), n1 >= 2 {
                size1D = n1
                continue
            }
            if keyword == "DOMAIN_MIN", parts.count >= 4,
               let r = Float(parts[1]), let g = Float(parts[2]), let b = Float(parts[3]) {
                domainMin = SIMD3<Float>(r, g, b)
                continue
            }
            if keyword == "DOMAIN_MAX", parts.count >= 4,
               let r = Float(parts[1]), let g = Float(parts[2]), let b = Float(parts[3]) {
                domainMax = SIMD3<Float>(r, g, b)
                continue
            }

            if parts.count >= 3,
               let r = Float(parts[0]),
               let g = Float(parts[1]),
               let b = Float(parts[2]) {
                rawTriplets.append((r, g, b))
            }
        }

        let expected3DCount = size3D * size3D * size3D
        if size3D > 0 && rawTriplets.count == size1D + expected3DCount && size1D > 0 {
            rawTriplets.removeFirst(size1D)
        }

        if size3D == 0 {
            let cbrtVal = Int(round(pow(Double(rawTriplets.count), 1.0 / 3.0)))
            if cbrtVal >= 2 && cbrtVal * cbrtVal * cbrtVal == rawTriplets.count {
                size3D = cbrtVal
            }
        }

        guard size3D >= 2, rawTriplets.count >= size3D * size3D * size3D else {
            throw NSError(domain: "CubeLUT", code: -1, userInfo: [NSLocalizedDescriptionKey: "无效的 .cube 3D LUT 文件（共解析 \(rawTriplets.count) 点）"])
        }

        var halfValues: [Float16] = []
        halfValues.reserveCapacity(size3D * size3D * size3D * 4)
        for i in 0..<(size3D * size3D * size3D) {
            let t = rawTriplets[i]
            halfValues.append(Float16(t.0))
            halfValues.append(Float16(t.1))
            halfValues.append(Float16(t.2))
            halfValues.append(Float16(1.0))
        }

        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rgba16Float
        desc.width = size3D
        desc.height = size3D
        desc.depth = size3D
        desc.usage = [.shaderRead]

        guard let texture = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "CubeLUT", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法创建 3D LUT GPU 纹理"])
        }

        halfValues.withUnsafeBytes { rawPtr in
            if let baseAddress = rawPtr.baseAddress {
                texture.replace(
                    region: MTLRegionMake3D(0, 0, 0, size3D, size3D, size3D),
                    mipmapLevel: 0,
                    slice: 0,
                    withBytes: baseAddress,
                    bytesPerRow: size3D * MemoryLayout<Float16>.size * 4,
                    bytesPerImage: size3D * size3D * MemoryLayout<Float16>.size * 4
                )
            }
        }

        return LoadedCubeLUT(
            texture: texture,
            size: size3D,
            name: url.deletingPathExtension().lastPathComponent,
            domainMin: domainMin,
            domainMax: domainMax
        )
    }

    static func exportCubeFile(lutParams: LUTGradingParameters) throws -> URL {
        let size = 33
        var lines: [String] = [
            "# Generated by CineLog 13 Pro Max",
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

    private static func accurateToneMapChannel(_ lin: Float) -> Float {
        let x: Float = max(0.0, lin) * 0.92
        let num: Float = x * (2.51 * x + 0.03)
        let den: Float = x * (2.43 * x + 0.59) + 0.14
        let acesLin: Float = clamp01(num / max(0.0001, den))
        let acesDisp: Float = pow(acesLin, 1.0 / 2.2)

        let comp: Float = clamp01(max(0.0, lin) / (1.0 + max(0.0, lin) * 0.28))
        let bt709: Float = (comp < 0.018) ? (4.5 * comp) : (1.099 * pow(comp, 0.45) - 0.099)
        return clamp01(bt709 * 0.45 + acesDisp * 0.55)
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

    private static func evaluateGradeCPU(appleLogRGB: SIMD3<Float>, u: LUTGradingParameters) -> SIMD3<Float> {
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

        let recR: Float = accurateToneMapChannel(linR709)
        let recG: Float = accurateToneMapChannel(linG709)
        let recB: Float = accurateToneMapChannel(linB709)

        let luma: Float = 0.2126 * recR + 0.7152 * recG + 0.0722 * recB
        let outR: Float = clamp01(luma + (recR - luma) * u.saturation)
        let outG: Float = clamp01(luma + (recG - luma) * u.saturation)
        let outB: Float = clamp01(luma + (recB - luma) * u.saturation)

        return SIMD3<Float>(outR, outG, outB)
    }
}

// MARK: - 4. Core 10-Bit Apple Log + Resident LUT + Full 4032x3024 Open Gate Camera Engine

final class AppleLogCameraEngine: NSObject, ObservableObject, @unchecked Sendable {
    enum EngineRecordingState {
        case idle
        case starting
        case recording
        case stopping
    }

    private static let kSavedResidentLUTFileNameKey = "CineLog13Pro.activeResidentLUTFileName"
    private static let kSavedLUTPresetRawKey        = "CineLog13Pro.savedLUTPresetRaw"
    private static let kSavedLUTInputSpaceRawKey    = "CineLog13Pro.savedLUTInputSpaceRaw"

    // Telemetry & Status
    @Published var isSessionRunning: Bool = false
    @Published var captureMode: CaptureModeOption = .video
    @Published var isRecording: Bool = false
    @Published var isCapturingPhoto: Bool = false
    @Published var showShutterFlash: Bool = false
    @Published var dualSaveRawAndGradedPhoto: Bool = false
    @Published var autoSaveVideoToPhotos: Bool = true
    @Published var tempStorageUsedMB: Double = 0.0
    @Published var memoryFootprintMB: Double = 0.0
    @Published var deviceFreeSpaceGB: Double = 0.0
    @Published var sessionSavedAssetIDs: [String] = []
    @Published var recordingDuration: TimeInterval = 0
    @Published var activePipelineBadge: String = "Initializing Cinema Pipeline..."
    @Published var activeResolutionBadge: String = "3840×2160 (10-Bit)"
    @Published var statusMessage: String = "Ready"
    @Published var lastRecordedPreviewFrame: UIImage?

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

    // Real-Time LUT Color Grading & Resident LUT Library
    @Published var lutSettings = LUTGradingParameters() {
        didSet {
            UserDefaults.standard.set(lutSettings.lutPreset.rawValue, forKey: Self.kSavedLUTPresetRawKey)
            UserDefaults.standard.set(lutSettings.lutInputSpace.rawValue, forKey: Self.kSavedLUTInputSpaceRawKey)
        }
    }
    @Published var residentLUTs: [ResidentLUTItem] = []
    @Published var activeResidentLUTFileName: String = ""
    @Published var bakeIntoRecording: Bool = false
    @Published var monitorDiagnostic: MonitorDiagnosticMode = .off
    @Published var aspectGuide: AspectGuideOption = .none
    @Published var showGridOverlay: Bool = true
    @Published var customLUTName: String = "未加载 .cube"

    var activeModeSummaryBadge: String {
        if lutSettings.lutPreset == .customCubeLUT {
            return "驻留LUT: \(customLUTName)"
        }
        return "LUT: \(lutSettings.lutPreset.title)"
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
    private(set) var dummy3DLUTTexture: MTLTexture?
    private(set) var custom3DLUTTexture: MTLTexture?
    private var custom3DLUTSize: Int = 33
    private var customLUTDomainMin = SIMD3<Float>(0.0, 0.0, 0.0)
    private var customLUTDomainMax = SIMD3<Float>(1.0, 1.0, 1.0)

    private let gpuFrameSemaphore = DispatchSemaphore(value: 2)
    private var recordBufferPool: CVPixelBufferPool?
    private var previewBufferPool: CVPixelBufferPool?
    private var currentRecordPoolDims = CMVideoDimensions(width: 0, height: 0)
    private var currentRecordPoolFormat: OSType = 0
    private var currentPreviewPoolDims = CMVideoDimensions(width: 0, height: 0)
    private var activeSensorDims = CMVideoDimensions(width: 3840, height: 2160)

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

    // Rock-Solid Atomic Recording State Machine with Watchdog Recovery
    private let recordingStateLock = NSLock()
    private var recordingState: EngineRecordingState = .idle
    private var lastStateTransitionTime: CFTimeInterval = 0
    private var assetWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var audioWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var recordingStartTime: CMTime?
    private var appendedFrameCount: Int = 0
    private var currentOutputURL: URL?
    private var activeRecordingUsesBGRA: Bool = false

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
        purgeAllTemporaryMediaFiles()
        setupMetalPipeline()
        restoreResidentLUTsOnLaunch()
        setupOrientationAndMotionObservers()
    }

    // MARK: - Resident 3D .cube LUT Library Management (Auto-Restores Across Launches!)

    private func restoreResidentLUTsOnLaunch() {
        self.residentLUTs = CubeLUTManager.listSavedResidentLUTs()

        if let savedInputRaw = UserDefaults.standard.object(forKey: Self.kSavedLUTInputSpaceRawKey) as? Int,
           let savedInput = LUTInputSpaceOption(rawValue: savedInputRaw) {
            self.lutSettings.lutInputSpace = savedInput
        }

        let savedFileName = UserDefaults.standard.string(forKey: Self.kSavedResidentLUTFileNameKey) ?? ""
        let targetItem = residentLUTs.first(where: { $0.fileName == savedFileName }) ?? residentLUTs.first

        if let item = targetItem, let device = metalDevice {
            if let loaded = try? CubeLUTManager.loadCubeLUT(from: item.fileURL, device: device) {
                self.custom3DLUTTexture = loaded.texture
                self.custom3DLUTSize = loaded.size
                self.customLUTDomainMin = loaded.domainMin
                self.customLUTDomainMax = loaded.domainMax
                self.activeResidentLUTFileName = item.fileName
                self.customLUTName = "\(loaded.name) (\(loaded.size)³)"
            }
        }

        if let savedPresetRaw = UserDefaults.standard.object(forKey: Self.kSavedLUTPresetRawKey) as? Int,
           let savedPreset = BuiltInLUTPreset(rawValue: savedPresetRaw) {
            if savedPreset == .customCubeLUT && self.custom3DLUTTexture == nil {
                self.lutSettings.lutPreset = .naturalRec709
            } else {
                self.lutSettings.lutPreset = savedPreset
            }
        } else if self.custom3DLUTTexture != nil {
            self.lutSettings.lutPreset = .customCubeLUT
        }
    }

    func importAndPersistResidentCubeLUT(from url: URL) {
        guard let device = metalDevice else { return }
        do {
            // 1. Validate & load 3D LUT texture first
            let loaded = try CubeLUTManager.loadCubeLUT(from: url, device: device)
            // 2. Save permanently to Documents/ResidentLUTs/
            let savedURL = try CubeLUTManager.saveToResidentLibrary(sourceURL: url)
            self.residentLUTs = CubeLUTManager.listSavedResidentLUTs()

            self.custom3DLUTTexture = loaded.texture
            self.custom3DLUTSize = loaded.size
            self.customLUTDomainMin = loaded.domainMin
            self.customLUTDomainMax = loaded.domainMax
            self.activeResidentLUTFileName = savedURL.lastPathComponent
            self.customLUTName = "\(loaded.name) (\(loaded.size)³)"
            self.lutSettings.lutPreset = .customCubeLUT

            UserDefaults.standard.set(savedURL.lastPathComponent, forKey: Self.kSavedResidentLUTFileNameKey)
            UserDefaults.standard.set(BuiltInLUTPreset.customCubeLUT.rawValue, forKey: Self.kSavedLUTPresetRawKey)

            self.statusMessage = "已设为驻留 LUT: \(loaded.name) (下次启动自动生效)"
        } catch {
            self.statusMessage = error.localizedDescription
        }
    }

    func selectResidentLUT(_ item: ResidentLUTItem) {
        guard let device = metalDevice else { return }
        do {
            let loaded = try CubeLUTManager.loadCubeLUT(from: item.fileURL, device: device)
            self.custom3DLUTTexture = loaded.texture
            self.custom3DLUTSize = loaded.size
            self.customLUTDomainMin = loaded.domainMin
            self.customLUTDomainMax = loaded.domainMax
            self.activeResidentLUTFileName = item.fileName
            self.customLUTName = "\(loaded.name) (\(loaded.size)³)"
            self.lutSettings.lutPreset = .customCubeLUT

            UserDefaults.standard.set(item.fileName, forKey: Self.kSavedResidentLUTFileNameKey)
            UserDefaults.standard.set(BuiltInLUTPreset.customCubeLUT.rawValue, forKey: Self.kSavedLUTPresetRawKey)
            self.statusMessage = "已切换驻留 LUT: \(loaded.name)"
        } catch {
            self.statusMessage = error.localizedDescription
        }
    }

    func deleteResidentLUT(_ item: ResidentLUTItem) {
        try? FileManager.default.removeItem(at: item.fileURL)
        self.residentLUTs = CubeLUTManager.listSavedResidentLUTs()

        if activeResidentLUTFileName == item.fileName {
            if let nextItem = residentLUTs.first {
                selectResidentLUT(nextItem)
            } else {
                custom3DLUTTexture = nil
                activeResidentLUTFileName = ""
                customLUTName = "未加载 .cube"
                lutSettings.lutPreset = .naturalRec709
                UserDefaults.standard.removeObject(forKey: Self.kSavedResidentLUTFileNameKey)
            }
        }
        statusMessage = "已移除驻留 LUT: \(item.displayName)"
    }

    // MARK: - Deep Storage, APFS Truncate-Before-Unlink & GPU/IOSurface Memory Purge

    /// Truncates a file to 0 bytes BEFORE unlinking so APFS immediately frees all disk extents
    /// even if a background system daemon (mediaserverd / assetsd) briefly holds a vnode reference.
    private func truncateAndRemoveFile(at url: URL) {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            if let handle = try? FileHandle(forWritingTo: url) {
                try? handle.truncate(atOffset: 0)
                try? handle.close()
            }
            try? fm.removeItem(at: url)
        }
    }

    /// Deeply purges NSTemporaryDirectory(), Library/Caches/, URLCache, CoreImage caches,
    /// CVMetalTextureCache, and idle CVPixelBufferPools (protecting Documents/ResidentLUTs/).
    func performDeepMemoryAndStoragePurge() {
        let beforeMem = Self.queryProcessMemoryMB()
        purgeAllTemporaryMediaFiles()

        videoProcessingQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.isRecording {
                if let recPool = self.recordBufferPool {
                    CVPixelBufferPoolFlush(recPool, .excessBuffers)
                }
                self.recordBufferPool = nil
                self.currentRecordPoolDims = CMVideoDimensions(width: 0, height: 0)
            }
            if let prevPool = self.previewBufferPool {
                CVPixelBufferPoolFlush(prevPool, .excessBuffers)
            }
            if let cache = self.textureCache {
                CVMetalTextureCacheFlush(cache, 0)
            }
            self.ciContext?.clearCaches()
            URLCache.shared.removeAllCachedResponses()

            let afterMem = Self.queryProcessMemoryMB()
            let freedMem = max(0.0, beforeMem - afterMem)
            self.refreshTempStorageUsage()
            DispatchQueue.main.async {
                self.statusMessage = String(format: "深度清理完成: 释放 %.1fMB 显存，缓存 0.0MB", freedMem)
            }
        }
    }

    func purgeAllTemporaryMediaFiles(keeping protectedURL: URL? = nil) {
        let fm = FileManager.default
        let residentPath = CubeLUTManager.residentLUTDirectoryURL.standardizedFileURL.path

        // 1. Clean NSTemporaryDirectory() recursively
        let tempDir = fm.temporaryDirectory
        if let enumerator = fm.enumerator(at: tempDir, includingPropertiesForKeys: [.isDirectoryKey], options: []) {
            for case let fileURL as URL in enumerator {
                if let keep = protectedURL, fileURL.standardizedFileURL == keep.standardizedFileURL {
                    continue
                }
                let isDir = (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if !isDir {
                    truncateAndRemoveFile(at: fileURL)
                }
            }
        }

        // 2. Clean Library/Caches/ temporary media & shader caches (counted by iOS as "System Data")
        if let cachesDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first,
           let cacheItems = try? fm.contentsOfDirectory(at: cachesDir, includingPropertiesForKeys: nil, options: []) {
            for itemURL in cacheItems {
                if itemURL.standardizedFileURL.path.hasPrefix(residentPath) { continue }
                try? fm.removeItem(at: itemURL)
            }
        }

        URLCache.shared.removeAllCachedResponses()
        refreshTempStorageUsage()
    }

    func deleteSessionVideosFromPhotoLibrary() {
        let ids = sessionSavedAssetIDs
        guard !ids.isEmpty else {
            statusMessage = "本次会话暂无已保存的测试视频"
            return
        }
        guard Bundle.main.object(forInfoDictionaryKey: "NSPhotoLibraryUsageDescription") != nil else {
            statusMessage = "请前往系统“照片 -> 最近删除”清空测试视频以释放闪存"
            return
        }
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
        guard fetchResult.count > 0 else {
            sessionSavedAssetIDs.removeAll()
            statusMessage = "测试视频已从相册移除"
            return
        }
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.deleteAssets(fetchResult)
        }) { [weak self] success, error in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if success {
                    self.sessionSavedAssetIDs.removeAll()
                    self.refreshTempStorageUsage()
                    self.statusMessage = "已从系统相册删除本次测试视频（请在系统相册“最近删除”中清空以彻底释放闪存）"
                } else {
                    self.statusMessage = "删除取消或失败: \(error?.localizedDescription ?? "")"
                }
            }
        }
    }

    static func queryProcessMemoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        let kr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            return Double(info.phys_footprint) / (1024.0 * 1024.0)
        }
        return 0.0
    }

    func refreshTempStorageUsage() {
        let fm = FileManager.default
        var totalBytes: Int64 = 0

        let tempDir = fm.temporaryDirectory
        if let enumerator = fm.enumerator(at: tempDir, includingPropertiesForKeys: [.fileSizeKey], options: []) {
            for case let fileURL as URL in enumerator {
                if let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                    totalBytes += Int64(size)
                }
            }
        }

        var freeGB: Double = 0.0
        if let homeURL = fm.urls(for: .documentDirectory, in: .userDomainMask).first,
           let vals = try? homeURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let avail = vals.volumeAvailableCapacityForImportantUsage {
            freeGB = Double(avail) / (1024.0 * 1024.0 * 1024.0)
        }

        let mb = Double(totalBytes) / (1024.0 * 1024.0)
        let memMB = Self.queryProcessMemoryMB()
        DispatchQueue.main.async {
            self.tempStorageUsedMB = mb
            self.memoryFootprintMB = memMB
            self.deviceFreeSpaceGB = freeGB
        }
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
        lutDesc.pixelFormat = .rgba16Float
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
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
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

    @objc private func handleAppDidEnterBackground() {
        if !isRecording {
            purgeAllTemporaryMediaFiles()
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
        switch preset {
        case .uhd4K16x9:
            candidatePool = fpsCompatibleFormats.filter {
                let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                let sub = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return d.width == 3840 && d.height == 2160 && tenBitSubtypes.contains(sub) && !$0.isVideoBinned
            }
        case .fhd1080p16x9:
            candidatePool = fpsCompatibleFormats.filter {
                let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                let sub = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return d.width == 1920 && d.height == 1080 && tenBitSubtypes.contains(sub)
            }
        case .openGateMax:
            let maxFourByThree = device.formats.filter { fmt in
                let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                let subType = CMFormatDescriptionGetMediaSubType(fmt.formatDescription)
                let ratio = Double(d.width) / Double(max(1, d.height))
                let supports24Or30 = fmt.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 24.0 }
                return allSupportedSubtypes.contains(subType) && abs(ratio - (4.0 / 3.0)) < 0.05 && d.width >= 2880 && supports24Or30
            }
            if !maxFourByThree.isEmpty {
                candidatePool = maxFourByThree
            } else {
                candidatePool = fpsCompatibleFormats.filter { fmt in
                    let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                    let ratio = Double(d.width) / Double(max(1, d.height))
                    return abs(ratio - (4.0 / 3.0)) < 0.05
                }
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
            if areaA != areaB {
                return areaA > areaB
            }
            let fullA = (subA == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) ? 1 : 0
            let fullB = (subB == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) ? 1 : 0
            return fullA > fullB
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

        let dims = CMVideoFormatDescriptionGetDimensions(activeFormat.formatDescription)
        self.activeSensorDims = dims
        self.hardwareSupportsNativeAppleLog = useNativeAppleLog
        self.activeFormatIsTenBit = isTenBit
        self.activeFormatIsFullRange = isFullRange
        self.activeFormatIsSDRFallback = isSDRInput

        let minISOVal = activeFormat.minISO
        let maxISOVal = min(activeFormat.maxISO, 3200.0)
        let currentISOVal = device.iso
        let bitStr = isTenBit ? "10-Bit BT.2020" : "Open Gate 4:3"

        DispatchQueue.main.async {
            self.minISO = minISOVal
            self.maxISO = maxISOVal
            self.currentISO = currentISOVal
            self.activeResolutionBadge = "\(dims.width)×\(dims.height) (\(bitStr))"
            self.activePipelineBadge = "\(self.selectedLens.focalLabel)"
            self.statusMessage = "管线就绪 (\(dims.width)×\(dims.height))"
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

            if self.dualSaveRawAndGradedPhoto, let rawTex = rawLogTex,
               let rawCI = self.makeOrientedCIImage(from: rawTex, colorSpace: colorSpace),
               let rawCG = ciCtx.createCGImage(rawCI, from: rawCI.extent, format: .RGBA16, colorSpace: colorSpace) {
                let rawUI = UIImage(cgImage: rawCG)
                self.savePhotoToLibrary(image: rawUI, label: "AppleLog_Raw")
            }
        }
    }

    private func makeOrientedCIImage(from texture: MTLTexture, colorSpace: CGColorSpace) -> CIImage? {
        guard var ciImg = CIImage(mtlTexture: texture, options: [.colorSpace: colorSpace]) else { return nil }
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

    // MARK: - Full-Size 4032x3024 Open Gate & 4K ProRes/HEVC Video Recording

    private func targetVideoDimensions(sensorW: Int, sensorH: Int, preset: SensorFormatPreset) -> (Int, Int) {
        return ((sensorW / 2) * 2, (sensorH / 2) * 2)
    }

    private func canHardwareEncode(width: Int, height: Int, codec: RecordingCodecOption) -> Bool {
        let cmCodec: CMVideoCodecType
        switch codec {
        case .proRes422HQ: cmCodec = kCMVideoCodecType_AppleProRes422HQ
        case .proRes422:   cmCodec = kCMVideoCodecType_AppleProRes422
        case .proRes422LT: cmCodec = kCMVideoCodecType_AppleProRes422LT
        case .hevc10Bit:   cmCodec = kCMVideoCodecType_HEVC
        }
        var session: VTCompressionSession?
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: Int32(width),
            height: Int32(height),
            codecType: cmCodec,
            encoderSpecification: nil,
            imageBufferAttributes: nil,
            compressedDataAllocator: nil,
            outputCallback: nil,
            refcon: nil,
            compressionSessionOut: &session
        )
        guard status == noErr, let validSession = session else { return false }
        let prepStatus = VTCompressionSessionPrepareToEncodeFrames(validSession)
        VTCompressionSessionInvalidate(validSession)
        return prepStatus == noErr
    }

    func triggerMainShutter() {
        if captureMode == .photo {
            captureStillPhoto()
        } else {
            toggleRecording()
        }
    }

    func toggleRecording() {
        let now = CACurrentMediaTime()
        recordingStateLock.lock()
        if (recordingState == .starting || recordingState == .stopping) && (now - lastStateTransitionTime > 2.0) {
            recordingState = isRecording ? .recording : .idle
        }
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
        lastStateTransitionTime = CACurrentMediaTime()
        recordingStateLock.unlock()

        let feedback = UIImpactFeedbackGenerator(style: .heavy)
        feedback.impactOccurred()

        videoProcessingQueue.async { [weak self] in
            guard let self = self else { return }

            self.purgeAllTemporaryMediaFiles()

            let sW = Int(self.activeSensorDims.width > 0 ? self.activeSensorDims.width : 3840)
            let sH = Int(self.activeSensorDims.height > 0 ? self.activeSensorDims.height : 2160)

            let (width, height) = self.targetVideoDimensions(sensorW: sW, sensorH: sH, preset: self.selectedSensorPreset)

            var chosenCodec = self.selectedCodec
            if self.selectedFPS == .fps60 && chosenCodec.isProRes && width > 1920 {
                chosenCodec = .hevc10Bit
            }
            if chosenCodec.isProRes && !self.canHardwareEncode(width: width, height: height, codec: chosenCodec) {
                chosenCodec = .hevc10Bit
            }

            let useBGRA = (chosenCodec == .hevc10Bit)
            self.activeRecordingUsesBGRA = useBGRA

            let recPoolFormat: OSType = useBGRA ? kCVPixelFormatType_32BGRA : kCVPixelFormatType_64RGBAHalf
            self.ensurePixelBufferPools(
                recordW: Int32(width),
                recordH: Int32(height),
                recordFormat: recPoolFormat,
                previewW: Int32(width),
                previewH: Int32(height)
            )

            let takeStr = String(format: "T%02d", self.clipTakeNumber)
            let fileName = "\(self.projectReelName)_\(takeStr)_\(width)x\(height)_\(Int(Date().timeIntervalSince1970)).mov"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try? FileManager.default.removeItem(at: outputURL)

            do {
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

                var videoSettings: [String: Any]
                if chosenCodec == .hevc10Bit {
                    let targetBitrate = (width * height > 9_000_000) ? 75_000_000 : ((width * height > 2_500_000) ? 55_000_000 : 22_000_000)
                    videoSettings = [
                        AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
                        AVVideoWidthKey: width,
                        AVVideoHeightKey: height,
                        AVVideoCompressionPropertiesKey: [
                            AVVideoAverageBitRateKey: targetBitrate,
                            AVVideoExpectedSourceFrameRateKey: Int(self.selectedFPS.rawValue)
                        ]
                    ]
                } else {
                    let isRawLogRec = !self.bakeIntoRecording || self.lutSettings.lutPreset == .rawAppleLog
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
                    videoSettings = [
                        AVVideoCodecKey: chosenCodec.avCodecType.rawValue,
                        AVVideoWidthKey: width,
                        AVVideoHeightKey: height,
                        AVVideoColorPropertiesKey: colorProperties
                    ]
                    if !writer.canApply(outputSettings: videoSettings, forMediaType: .video) {
                        chosenCodec = .hevc10Bit
                        self.activeRecordingUsesBGRA = true
                        self.ensurePixelBufferPools(
                            recordW: Int32(width),
                            recordH: Int32(height),
                            recordFormat: kCVPixelFormatType_32BGRA,
                            previewW: Int32(width),
                            previewH: Int32(height)
                        )
                        videoSettings = [
                            AVVideoCodecKey: AVVideoCodecType.hevc.rawValue,
                            AVVideoWidthKey: width,
                            AVVideoHeightKey: height,
                            AVVideoCompressionPropertiesKey: [
                                AVVideoAverageBitRateKey: 55_000_000,
                                AVVideoExpectedSourceFrameRateKey: Int(self.selectedFPS.rawValue)
                            ]
                        ]
                    }
                }

                let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
                vInput.expectsMediaDataInRealTime = true
                vInput.transform = self.orientationMode.assetWriterTransform

                let adaptorFormat: OSType = self.activeRecordingUsesBGRA ? kCVPixelFormatType_32BGRA : kCVPixelFormatType_64RGBAHalf
                let sourcePixelAttributes: [String: Any] = [
                    kCVPixelBufferPixelFormatTypeKey as String: adaptorFormat,
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
                self.appendedFrameCount = 0
                self.currentOutputURL = outputURL

                self.recordingStateLock.lock()
                self.recordingState = .recording
                self.lastStateTransitionTime = CACurrentMediaTime()
                self.recordingStateLock.unlock()

                let modeDesc = self.bakeIntoRecording ? "LUT直出 (\(self.activeModeSummaryBadge))" : "纯净 Apple Log"
                DispatchQueue.main.async {
                    self.selectedCodec = chosenCodec
                    self.isRecording = true
                    self.recordingDuration = 0
                    self.statusMessage = "REC \(width)×\(height) \(chosenCodec.rawValue) • \(modeDesc)"
                }
            } catch {
                try? FileManager.default.removeItem(at: outputURL)
                self.recordingStateLock.lock()
                self.recordingState = .idle
                self.lastStateTransitionTime = CACurrentMediaTime()
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
            lastStateTransitionTime = CACurrentMediaTime()
            recordingStateLock.unlock()
            DispatchQueue.main.async { self.isRecording = false }
            return
        }
        recordingState = .stopping
        lastStateTransitionTime = CACurrentMediaTime()
        recordingStateLock.unlock()

        let feedback = UIImpactFeedbackGenerator(style: .medium)
        feedback.impactOccurred()

        videoProcessingQueue.asyncAfter(deadline: .now() + 0.08) { [weak self] in
            guard let self = self else { return }

            guard let writer = self.assetWriter else {
                self.resetRecordingStateToIdle()
                return
            }

            let outputURL = self.currentOutputURL

            if writer.status != .writing || self.recordingStartTime == nil || self.appendedFrameCount == 0 {
                writer.cancelWriting()
                if let url = outputURL {
                    self.truncateAndRemoveFile(at: url)
                }
                self.resetRecordingStateToIdle(message: "录制时间过短，已清理临时缓存")
                self.refreshTempStorageUsage()
                return
            }

            self.videoWriterInput?.markAsFinished()
            self.audioWriterInput?.markAsFinished()

            writer.finishWriting {
                // Capture preview frame directly from GPU texture in RAM (never open the .mov with AVAssetImageGenerator!)
                self.cachePreviewFrameFromTexture()
                self.resetRecordingStateToIdle(incrementTake: true)
                if let url = outputURL {
                    if self.autoSaveVideoToPhotos {
                        self.saveVideoToPhotoLibraryAndDeleteTemp(url: url)
                    } else {
                        self.truncateAndRemoveFile(at: url)
                        self.refreshTempStorageUsage()
                        DispatchQueue.main.async {
                            self.statusMessage = "未保存视频，已自动截断并释放全部闪存块"
                        }
                    }
                }
            }
        }
    }

    private func cachePreviewFrameFromTexture() {
        previewLock.lock()
        let tex = _latestRecordTexture ?? _latestPreviewTexture
        previewLock.unlock()
        guard let validTex = tex, let ciCtx = ciContext else { return }
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        if let ciImg = makeOrientedCIImage(from: validTex, colorSpace: cs) {
            let scale = min(1.0, 1280.0 / max(1.0, max(ciImg.extent.width, ciImg.extent.height)))
            let scaled = ciImg.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            if let cgImg = ciCtx.createCGImage(scaled, from: scaled.extent, format: .RGBA8, colorSpace: cs) {
                let uiImg = UIImage(cgImage: cgImg)
                DispatchQueue.main.async {
                    self.lastRecordedPreviewFrame = uiImg
                }
            }
        }
    }

    private func resetRecordingStateToIdle(incrementTake: Bool = false, message: String? = nil) {
        self.assetWriter = nil
        self.videoWriterInput = nil
        self.audioWriterInput = nil
        self.pixelBufferAdaptor = nil
        self.recordingStartTime = nil
        self.appendedFrameCount = 0
        self.currentOutputURL = nil

        // Immediately release the recording pixel buffer pool & flush Metal texture cache to free 200-400MB of IOSurface RAM!
        if let recPool = self.recordBufferPool {
            CVPixelBufferPoolFlush(recPool, .excessBuffers)
        }
        self.recordBufferPool = nil
        self.currentRecordPoolDims = CMVideoDimensions(width: 0, height: 0)
        self._latestRecordTexture = nil
        if let cache = self.textureCache {
            CVMetalTextureCacheFlush(cache, 0)
        }

        recordingStateLock.lock()
        recordingState = .idle
        lastStateTransitionTime = CACurrentMediaTime()
        recordingStateLock.unlock()

        DispatchQueue.main.async {
            self.isRecording = false
            if incrementTake {
                self.clipTakeNumber += 1
            }
            if let msg = message {
                self.statusMessage = msg
            } else {
                self.statusMessage = "正在保存视频并释放临时缓存..."
            }
        }
    }

    private func saveVideoToPhotoLibraryAndDeleteTemp(url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                self.truncateAndRemoveFile(at: url)
                self.refreshTempStorageUsage()
                DispatchQueue.main.async {
                    self.statusMessage = "相册权限未开启，已清理临时视频缓存"
                }
                return
            }
            var createdAssetID: String?
            PHPhotoLibrary.shared().performChanges({
                // Use PHAssetCreationRequest with shouldMoveFile = true so iOS moves the file atomically in APFS
                // without creating a duplicate multi-GB staging copy in "System Data (系统数据)"!
                let creationReq = PHAssetCreationRequest.forAsset()
                let moveOpts = PHAssetResourceCreationOptions()
                moveOpts.shouldMoveFile = true
                creationReq.addResource(with: .video, fileURL: url, options: moveOpts)
                createdAssetID = creationReq.placeholderForCreatedAsset?.localIdentifier
            }) { success, error in
                self.truncateAndRemoveFile(at: url)
                self.refreshTempStorageUsage()
                DispatchQueue.main.async {
                    if success {
                        if let id = createdAssetID {
                            self.sessionSavedAssetIDs.append(id)
                        }
                        self.statusMessage = "已零拷贝移入相册 (\(self.selectedCodec.rawValue)) • 沙盒缓存 0MB"
                    } else {
                        self.statusMessage = "保存失败 (\(error?.localizedDescription ?? "未知"))，已释放缓存"
                    }
                }
            }
        }
    }

    private func ensurePixelBufferPools(
        recordW: Int32,
        recordH: Int32,
        recordFormat: OSType,
        previewW: Int32,
        previewH: Int32
    ) {
        let evenRecW = (recordW / 2) * 2
        let evenRecH = (recordH / 2) * 2
        let evenPrevW = (previewW / 2) * 2
        let evenPrevH = (previewH / 2) * 2

        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 3]

        if currentRecordPoolDims.width != evenRecW ||
           currentRecordPoolDims.height != evenRecH ||
           currentRecordPoolFormat != recordFormat ||
           recordBufferPool == nil {
            currentRecordPoolDims = CMVideoDimensions(width: evenRecW, height: evenRecH)
            currentRecordPoolFormat = recordFormat
            let recAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: recordFormat,
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

// MARK: - 5. AVCaptureVideoDataOutput Delegate

extension AppleLogCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

    struct PipelineUniforms {
        var isNativeAppleLog: Int32
        var isOpenGateMode: Int32
        var isFullRangeYCbCr: Int32
        var isTenBitSensorInput: Int32

        var isSDRSensorInput: Int32
        var writeRecordBuffer: Int32
        var bakeLUTIntoRecording: Int32
        var monitorDiagnosticMode: Int32

        var focusPeakingEnabled: Int32
        var focusPeakingColorIndex: Int32
        var focusPeakingThreshold: Float
        var lutPresetID: Int32

        var lutInputSpace: Int32
        var customLUTSize: Int32
        var hasCustom3DLUT: Int32
        var lutIntensity: Float

        var exposureEV: Float
        var contrast: Float
        var highlightRollOff: Float
        var shadowLift: Float

        var saturation: Float
        var vibrance: Float
        var temperature: Float
        var tint: Float

        var debandingStrength: Float
        var frameTime: Float
        var pad0: Float = 0.0
        var pad1: Float = 0.0

        var domainMin: SIMD4<Float>
        var domainMax: SIMD4<Float>
        var liftRGB: SIMD4<Float>
        var gammaRGB: SIMD4<Float>
        var gainRGB: SIMD4<Float>
    }

    func makeShaderUniforms(writeRecord: Bool) -> PipelineUniforms {
        let l = lutSettings
        return PipelineUniforms(
            isNativeAppleLog: hardwareSupportsNativeAppleLog ? 1 : 0,
            isOpenGateMode: (selectedSensorPreset == .openGateMax) ? 1 : 0,
            isFullRangeYCbCr: activeFormatIsFullRange ? 1 : 0,
            isTenBitSensorInput: activeFormatIsTenBit ? 1 : 0,
            isSDRSensorInput: activeFormatIsSDRFallback ? 1 : 0,
            writeRecordBuffer: writeRecord ? 1 : 0,
            bakeLUTIntoRecording: bakeIntoRecording ? 1 : 0,
            monitorDiagnosticMode: Int32(monitorDiagnostic.rawValue),
            focusPeakingEnabled: focusPeakingEnabled ? 1 : 0,
            focusPeakingColorIndex: Int32(focusPeakingColor.rawValue),
            focusPeakingThreshold: focusPeakingThreshold,
            lutPresetID: Int32(l.lutPreset.rawValue),
            lutInputSpace: Int32(l.lutInputSpace.rawValue),
            customLUTSize: Int32(custom3DLUTSize),
            hasCustom3DLUT: custom3DLUTTexture != nil ? 1 : 0,
            lutIntensity: l.lutIntensity,
            exposureEV: l.exposureEV,
            contrast: l.contrast,
            highlightRollOff: l.highlightRollOff,
            shadowLift: l.shadowLift,
            saturation: l.saturation,
            vibrance: l.vibrance,
            temperature: l.temperature,
            tint: l.tint,
            debandingStrength: l.debanding,
            frameTime: Float(CACurrentMediaTime() - startTimestamp),
            domainMin: SIMD4<Float>(customLUTDomainMin.x, customLUTDomainMin.y, customLUTDomainMin.z, 0.0),
            domainMax: SIMD4<Float>(customLUTDomainMax.x, customLUTDomainMax.y, customLUTDomainMax.z, 1.0),
            liftRGB: SIMD4<Float>(l.liftR, l.liftG, l.liftB, 0.0),
            gammaRGB: SIMD4<Float>(l.gammaR, l.gammaG, l.gammaB, 0.0),
            gainRGB: SIMD4<Float>(l.gainR, l.gainG, l.gainB, 1.0)
        )
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if output == audioDataOutput {
            updateAudioPeakLevels(from: sampleBuffer)
            recordingStateLock.lock()
            let recActive = (recordingState == .recording || recordingState == .stopping)
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
        self.activeSensorDims = CMVideoDimensions(width: Int32(sensorW), height: Int32(sensorH))

        recordingStateLock.lock()
        let recActive = (recordingState == .recording || recordingState == .stopping)
        recordingStateLock.unlock()

        previewLock.lock()
        let isPhotoSnap = pendingPhotoCaptureRequest
        previewLock.unlock()

        let needRecordOut = recActive || isPhotoSnap

        let computeW: Int
        let computeH: Int
        if selectedSensorPreset == .openGateMax {
            if isPhotoSnap && !recActive {
                computeW = sensorW
                computeH = sensorH
            } else if recActive {
                let (safeW, safeH) = targetVideoDimensions(sensorW: sensorW, sensorH: sensorH, preset: .openGateMax)
                computeW = safeW
                computeH = safeH
            } else {
                computeW = (sensorW / 4) * 2
                computeH = (sensorH / 4) * 2
            }
        } else {
            computeW = sensorW
            computeH = sensorH
        }

        let recPoolFormat: OSType = (recActive && activeRecordingUsesBGRA)
            ? kCVPixelFormatType_32BGRA
            : kCVPixelFormatType_64RGBAHalf
        let recMetalFormat: MTLPixelFormat = (recPoolFormat == kCVPixelFormatType_32BGRA)
            ? .bgra8Unorm
            : .rgba16Float

        ensurePixelBufferPools(
            recordW: Int32(computeW),
            recordH: Int32(computeH),
            recordFormat: recPoolFormat,
            previewW: Int32(computeW),
            previewH: Int32(computeH)
        )

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
        if needRecordOut {
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, recPool, &logOutPixelBuffer)
        }
        let logOutPB = logOutPixelBuffer ?? prevOutPB

        var cvLogOutTex: CVMetalTexture?
        var cvPrevOutTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, prevOutPB, nil, .rgba16Float, computeW, computeH, 0, &cvPrevOutTex
        )
        if needRecordOut {
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, cache, logOutPB, nil, recMetalFormat, computeW, computeH, 0, &cvLogOutTex
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
        var uniforms = makeShaderUniforms(writeRecord: needRecordOut)

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
            if needRecordOut {
                self._latestRecordTexture = logOutTex
            }
            self.previewLock.unlock()

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
            CVMetalTextureCacheFlush(cache, 0)
        }
        if frameCounter % 120 == 0 {
            refreshTempStorageUsage()
        }
    }

    private func appendRecordedPixelBuffer(_ pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        recordingStateLock.lock()
        let canAppend = (recordingState == .recording || (recordingState == .stopping && appendedFrameCount == 0))
        recordingStateLock.unlock()

        guard canAppend, let writer = assetWriter else { return }
        let pbW = Int32(CVPixelBufferGetWidth(pixelBuffer))
        let pbH = Int32(CVPixelBufferGetHeight(pixelBuffer))
        guard pbW == currentRecordPoolDims.width && pbH == currentRecordPoolDims.height else { return }

        if writer.status == .failed {
            let errMsg = writer.error?.localizedDescription ?? "编码器写入异常"
            if let url = currentOutputURL {
                try? FileManager.default.removeItem(at: url)
            }
            resetRecordingStateToIdle(message: "录制异常中断: \(errMsg)")
            refreshTempStorageUsage()
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
                if adaptor.append(pixelBuffer, withPresentationTime: timestamp) {
                    appendedFrameCount += 1
                } else {
                    let errMsg = writer.error?.localizedDescription ?? "像素缓冲写入失败"
                    if let url = currentOutputURL {
                        try? FileManager.default.removeItem(at: url)
                    }
                    resetRecordingStateToIdle(message: "编码中止: \(errMsg)")
                    refreshTempStorageUsage()
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
            for y in stride(from: 0, to: height, by: 32) {
                let rowPtr = baseAddr.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt16.self)
                for x in stride(from: 0, to: width, by: 32) {
                    let val10 = Int(rowPtr[x] >> 6)
                    let norm = max(0, min(63, (val10 - 64) * 63 / max(1, 940 - 64)))
                    bins[norm] += 1
                    totalSamples += 1
                }
            }
        } else {
            for y in stride(from: 0, to: height, by: 32) {
                let rowPtr = baseAddr.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
                for x in stride(from: 0, to: width, by: 32) {
                    let val8 = Int(rowPtr[x])
                    let norm = max(0, min(63, val8 * 63 / 255))
                    bins[norm] += 1
                    totalSamples += 1
                }
            }
        }

        if totalSamples > 0 {
            let maxBin = max(bins.max() ?? 1.0, 1.0)
            for i in 0..<64 {
                bins[i] /= maxBin
            }
        }

        DispatchQueue.main.async {
            self.lumaHistogram = bins
        }
    }
}

// MARK: - 6. Calibrated sRGB Event-Driven Metal Viewfinder

struct MetalLogViewfinder: UIViewRepresentable {
    @ObservedObject var engine: AppleLogCameraEngine

    func makeCoordinator() -> Coordinator {
        Coordinator(engine: engine)
    }

    func makeUIView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: engine.metalDevice)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.framebufferOnly = true
        mtkView.clearColor = MTLClearColor(red: 0.03, green: 0.03, blue: 0.03, alpha: 1.0)
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true
        mtkView.autoResizeDrawable = true
        mtkView.contentMode = .scaleAspectFit

        if let metalLayer = mtkView.layer as? CAMetalLayer {
            metalLayer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
            metalLayer.wantsExtendedDynamicRangeContent = false
        }

        engine.mtkViewRef = mtkView
        mtkView.delegate = context.coordinator
        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        uiView.setNeedsDisplay()
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        let engine: AppleLogCameraEngine
        private var renderPipelineState: MTLRenderPipelineState?
        private var commandQueue: MTLCommandQueue?

        struct ViewfinderVertexUniforms {
            var scale: SIMD2<Float>
            var orientationMode: Int32
            var mirrorHorizontal: Int32
        }

        init(engine: AppleLogCameraEngine) {
            self.engine = engine
            super.init()
            guard let device = engine.metalDevice,
                  let library = engine.metalLibrary else { return }
            self.commandQueue = device.makeCommandQueue()

            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = library.makeFunction(name: "vertexViewfinder")
            desc.fragmentFunction = library.makeFunction(name: "fragmentViewfinder")
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.renderPipelineState = try? device.makeRenderPipelineState(descriptor: desc)
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let passDescriptor = view.currentRenderPassDescriptor,
                  let previewTex = engine.latestPreviewTexture,
                  let pipeline = renderPipelineState,
                  let queue = commandQueue,
                  let commandBuffer = queue.makeCommandBuffer(),
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else {
                return
            }

            let viewW = Float(max(1.0, view.drawableSize.width))
            let viewH = Float(max(1.0, view.drawableSize.height))
            let isPortrait = engine.orientationMode.isPortraitSwapped
            let texW = Float(isPortrait ? previewTex.height : previewTex.width)
            let texH = Float(isPortrait ? previewTex.width : previewTex.height)

            let viewAspect = viewW / viewH
            let texAspect = texW / max(1.0, texH)

            var scale = SIMD2<Float>(1.0, 1.0)
            if texAspect > viewAspect {
                scale.y = viewAspect / texAspect
            } else {
                scale.x = texAspect / viewAspect
            }

            var vUniforms = ViewfinderVertexUniforms(
                scale: scale,
                orientationMode: Int32(engine.orientationMode.rawValue),
                mirrorHorizontal: engine.isMirroredHorizontal ? 1 : 0
            )

            renderEncoder.setRenderPipelineState(pipeline)
            renderEncoder.setVertexBytes(&vUniforms, length: MemoryLayout<ViewfinderVertexUniforms>.stride, index: 0)
            renderEncoder.setFragmentTexture(previewTex, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.endEncoding()

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}

// MARK: - 7. Post-Production Non-Destructive Apple Log Video / Photo Grading Studio

final class PostProductionLogGrader: ObservableObject {
    @Published var sourceImage: UIImage?
    @Published var gradedPreviewImage: UIImage?
    @Published var exportStatus: String = ""

    private let ciContext = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB) as Any])

    func loadCachedPreviewFrame(from engine: AppleLogCameraEngine) {
        if let cached = engine.lastRecordedPreviewFrame {
            self.sourceImage = cached
            self.renderGradedStill(engine: engine)
        }
    }

    func extractPreviewFrame(from videoURL: URL, engine: AppleLogCameraEngine) {
        let asset = AVAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        if let cgImg = try? generator.copyCGImage(at: CMTime(seconds: 0.1, preferredTimescale: 600), actualTime: nil) {
            let uiImg = UIImage(cgImage: cgImg)
            DispatchQueue.main.async {
                self.sourceImage = uiImg
                self.renderGradedStill(engine: engine)
            }
        }
    }

    func renderGradedStill(engine: AppleLogCameraEngine) {
        guard let srcUI = sourceImage,
              let cgIn = srcUI.cgImage,
              let device = engine.metalDevice,
              let postPipeline = engine.postGradePipelineState,
              let queue = device.makeCommandQueue(),
              let activeLUT = engine.custom3DLUTTexture ?? engine.dummy3DLUTTexture else { return }

        let loader = MTKTextureLoader(device: device)
        guard let inTex = try? loader.newTexture(cgImage: cgIn, options: [.SRGB: false]) else { return }

        let outDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float,
            width: inTex.width,
            height: inTex.height,
            mipmapped: false
        )
        outDesc.usage = [.shaderWrite, .shaderRead]
        guard let outTex = device.makeTexture(descriptor: outDesc),
              let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        var uniforms = engine.makeShaderUniforms(writeRecord: false)
        enc.setComputePipelineState(postPipeline)
        enc.setTexture(inTex, index: 0)
        enc.setTexture(outTex, index: 1)
        enc.setTexture(activeLUT, index: 2)
        enc.setBytes(&uniforms, length: MemoryLayout<AppleLogCameraEngine.PipelineUniforms>.stride, index: 0)

        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(
            width: (inTex.width + 15) / 16,
            height: (inTex.height + 15) / 16,
            depth: 1
        )
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        if var ciOut = CIImage(mtlTexture: outTex, options: [.colorSpace: cs]) {
            ciOut = ciOut.transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -ciOut.extent.height))
            if let cgOut = ciContext.createCGImage(ciOut, from: ciOut.extent, format: .RGBA8, colorSpace: cs) {
                let finalImg = UIImage(cgImage: cgOut)
                DispatchQueue.main.async {
                    self.gradedPreviewImage = finalImg
                }
            }
        }
    }

    func saveGradedPhotoToLibrary() {
        guard let img = gradedPreviewImage else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: img)
            }) { success, _ in
                DispatchQueue.main.async {
                    self.exportStatus = success ? "已保存调色帧至系统相册" : "保存失败"
                }
            }
        }
    }
}

// MARK: - 8. Blackmagic-Style HUD Overlays

struct GyroHorizonOverlayView: View {
    let rollDegrees: Double
    let pitchDegrees: Double

    var isLevel: Bool {
        abs(rollDegrees) < 1.0
    }

    var body: some View {
        ZStack {
            HStack(spacing: 40) {
                Rectangle()
                    .fill(Color.white.opacity(0.35))
                    .frame(width: 24, height: 1.5)
                Rectangle()
                    .fill(Color.white.opacity(0.35))
                    .frame(width: 24, height: 1.5)
            }

            Rectangle()
                .fill(isLevel ? Color.green : Color.yellow.opacity(0.85))
                .frame(width: 90, height: isLevel ? 2.0 : 1.5)
                .rotationEffect(.degrees(-rollDegrees))

            Circle()
                .stroke(isLevel ? Color.green : Color.yellow.opacity(0.85), lineWidth: 1.5)
                .frame(width: 10, height: 10)

            Text(String(format: "%+.1f°", rollDegrees))
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(isLevel ? .green : .yellow)
                .offset(y: 16)
        }
        .allowsHitTesting(false)
    }
}

struct StereoAudioVUMeterView: View {
    let leftDB: Float
    let rightDB: Float

    private func normalizedLevel(_ db: Float) -> CGFloat {
        CGFloat(max(0.0, min(1.0, (db + 60.0) / 60.0)))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 3) {
                Text("L")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.15))
                        Capsule()
                            .fill(leftDB > -6.0 ? Color.red : (leftDB > -18.0 ? Color.yellow : Color.green))
                            .frame(width: geo.size.width * normalizedLevel(leftDB))
                    }
                }
            }
            .frame(height: 5)

            HStack(spacing: 3) {
                Text("R")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.15))
                        Capsule()
                            .fill(rightDB > -6.0 ? Color.red : (rightDB > -18.0 ? Color.yellow : Color.green))
                            .frame(width: geo.size.width * normalizedLevel(rightDB))
                    }
                }
            }
            .frame(height: 5)
        }
        .frame(width: 56, height: 16)
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.18), lineWidth: 1))
        .fixedSize()
    }
}

struct LumaHistogramOverlay: View {
    let bins: [Float]

    var body: some View {
        HStack(alignment: .bottom, spacing: 1) {
            ForEach(0..<bins.count, id: \.self) { idx in
                Rectangle()
                    .fill(barColor(for: idx))
                    .frame(width: 1.15, height: max(2, CGFloat(bins[idx]) * 18))
            }
        }
        .frame(width: 96, height: 20)
        .padding(4)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.2), lineWidth: 1))
        .fixedSize()
    }

    private func barColor(for index: Int) -> Color {
        if index > 58 { return .red }
        if index >= 26 && index <= 34 { return .green }
        return .white.opacity(0.85)
    }
}

struct AspectGuideAndGridOverlayView: View {
    let guide: AspectGuideOption
    let showGrid: Bool
    let focusReticle: CGPoint?
    let isLocked: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack {
                if showGrid {
                    Path { path in
                        let w = geo.size.width
                        let h = geo.size.height
                        path.move(to: CGPoint(x: w / 3, y: 0))
                        path.addLine(to: CGPoint(x: w / 3, y: h))
                        path.move(to: CGPoint(x: 2 * w / 3, y: 0))
                        path.addLine(to: CGPoint(x: 2 * w / 3, y: h))
                        path.move(to: CGPoint(x: 0, y: h / 3))
                        path.addLine(to: CGPoint(x: w, y: h / 3))
                        path.move(to: CGPoint(x: 0, y: 2 * h / 3))
                        path.addLine(to: CGPoint(x: w, y: 2 * h / 3))
                    }
                    .stroke(Color.white.opacity(0.22), lineWidth: 0.7)
                }

                if guide != .none {
                    let rect = guideRect(in: geo.size)
                    Rectangle()
                        .stroke(Color.yellow.opacity(0.75), lineWidth: 1.2)
                        .frame(width: rect.width, height: rect.height)
                }

                if let pt = focusReticle {
                    ZStack {
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(isLocked ? Color.yellow : Color.green, lineWidth: 1.8)
                            .frame(width: 54, height: 54)
                        if isLocked {
                            Image(systemName: "lock.fill")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundColor(.yellow)
                                .offset(y: -36)
                        }
                    }
                    .position(x: pt.x * geo.size.width, y: pt.y * geo.size.height)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func guideRect(in size: CGSize) -> CGRect {
        let targetAspect: CGFloat
        switch guide {
        case .none:        return CGRect(origin: .zero, size: size)
        case .scope239:    targetAspect = 2.39
        case .cinema185:   targetAspect = 1.85
        case .vertical916: targetAspect = 9.0 / 16.0
        }
        let viewAspect = size.width / max(1, size.height)
        if targetAspect > viewAspect {
            let h = size.width / targetAspect
            return CGRect(x: 0, y: (size.height - h) / 2, width: size.width, height: h)
        } else {
            let w = size.height * targetAspect
            return CGRect(x: (size.width - w) / 2, y: 0, width: w, height: size.height)
        }
    }
}

// MARK: - 9. Symmetrical & Width-Locked Adaptive Cinema Camera UI (Portrait & Landscape)

enum ActiveControlDrawer: Int {
    case none       = 0
    case format     = 1
    case focus      = 2
    case exposure   = 3
    case lutGrading = 4
    case monitor    = 5
}

struct CineLogMainCameraView: View {
    @StateObject private var engine = AppleLogCameraEngine()
    @StateObject private var postGrader = PostProductionLogGrader()

    @State private var activeTab: ActiveControlDrawer = .none
    @State private var showCubeImporter: Bool = false
    @State private var showPostGradingSheet: Bool = false
    @State private var exportedCubeURL: URL?
    @State private var showCubeShareSheet: Bool = false

    var body: some View {
        GeometryReader { geo in
            let screenW = geo.size.width
            let screenH = geo.size.height
            let isLandscape = screenW > screenH
            let cardWidth = max(280, screenW - 20)

            ZStack(alignment: .center) {
                Color.black.ignoresSafeArea()

                // 1. Live Metal Viewfinder
                MetalLogViewfinder(engine: engine)
                    .frame(width: screenW, height: screenH)
                    .ignoresSafeArea()
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onEnded { value in
                                if activeTab != .none && isLandscape {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        activeTab = .none
                                    }
                                    return
                                }
                                let normX = max(0.05, min(0.95, value.location.x / max(1, screenW)))
                                let normY = max(0.05, min(0.95, value.location.y / max(1, screenH)))
                                engine.handleViewfinderTap(normalizedPoint: CGPoint(x: normX, y: normY), lockFocusAndExposure: false)
                            }
                    )
                    .simultaneousGesture(
                        LongPressGesture(minimumDuration: 0.5)
                            .onEnded { _ in
                                if let pt = engine.focusReticleNormalized {
                                    engine.handleViewfinderTap(normalizedPoint: pt, lockFocusAndExposure: true)
                                }
                            }
                    )

                // 2. Grid, Aspect Ratio Guide & Focus Reticle Overlay
                AspectGuideAndGridOverlayView(
                    guide: engine.aspectGuide,
                    showGrid: engine.showGridOverlay,
                    focusReticle: engine.focusReticleNormalized,
                    isLocked: engine.isFocusLockedByTap
                )
                .frame(width: screenW, height: screenH)
                .ignoresSafeArea()

                // 3. Gyro Horizon Leveler
                if engine.showHorizonLeveler {
                    GyroHorizonOverlayView(
                        rollDegrees: engine.deviceRollDegrees,
                        pitchDegrees: engine.devicePitchDegrees
                    )
                }

                // 4. Still Photo Shutter Flash Overlay
                if engine.showShutterFlash {
                    Color.white
                        .opacity(0.75)
                        .ignoresSafeArea()
                        .allowsHitTesting(false)
                }

                // 5. Width-Locked HUD Container (Strictly Clamped to screenW so Zero Horizontal Drift Occurs!)
                VStack(spacing: isLandscape ? 3 : 5) {
                    if isLandscape {
                        // === LANDSCAPE COMPACT TOP BAR ===
                        HStack(spacing: 6) {
                            Button {
                                engine.rotateOrientation90Degrees()
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: engine.orientationMode.isPortraitSwapped ? "rectangle.portrait" : "rectangle")
                                    Text(engine.orientationMode.shortLabel)
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.70))
                                .foregroundColor(.yellow)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.yellow.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            Button {
                                showPostGradingSheet = true
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: "slider.horizontal.below.rectangle")
                                    Text("调色室")
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(Color.cyan.opacity(0.25))
                                .foregroundColor(.cyan)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            Button {
                                engine.performDeepMemoryAndStoragePurge()
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: "bolt.shield.fill")
                                    Text("清缓存")
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(Color.orange.opacity(0.25))
                                .foregroundColor(.orange)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.orange.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 5) {
                                    TelemetryPill(label: "RES", value: engine.selectedSensorPreset.shortBadge, highlight: true)
                                    TelemetryPill(label: "FPS", value: "\(Int(engine.selectedFPS.rawValue))", highlight: false)
                                    TelemetryPill(label: "SHUTTER", value: "\(Int(engine.shutterAngleDegrees))°", highlight: engine.isManualExposure)
                                    TelemetryPill(label: "ISO", value: "\(Int(engine.currentISO))", highlight: engine.isManualExposure)
                                    TelemetryPill(label: "WB", value: "\(Int(engine.whiteBalanceKelvin))K", highlight: engine.isWhiteBalanceLocked)
                                    TelemetryPill(label: "FOCUS", value: "\(engine.focusMode.shortTitle) \(String(format: "%.2f", engine.lensPosition))", highlight: engine.focusMode == .manualFocus)
                                    TelemetryPill(label: "STAB", value: engine.stabilizationMode.shortTitle, highlight: engine.stabilizationMode != .off)
                                }
                            }

                            StereoAudioVUMeterView(leftDB: engine.audioPeakDBLeft, rightDB: engine.audioPeakDBRight)
                            LumaHistogramOverlay(bins: engine.lumaHistogram)
                        }
                        .padding(.horizontal, 10)
                        .padding(.top, 4)
                        .frame(width: screenW)
                        .layoutPriority(10)
                    } else {
                        // === PORTRAIT TOP HUD (Strictly Width-Clamped to screenW!) ===
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 5) {
                                TelemetryPill(label: "RES", value: engine.selectedSensorPreset.shortBadge, highlight: true)
                                TelemetryPill(label: "FPS", value: "\(Int(engine.selectedFPS.rawValue))", highlight: false)
                                TelemetryPill(label: "SHUTTER", value: "\(Int(engine.shutterAngleDegrees))°", highlight: engine.isManualExposure)
                                TelemetryPill(label: "ISO", value: "\(Int(engine.currentISO))", highlight: engine.isManualExposure)
                                TelemetryPill(label: "WB", value: "\(Int(engine.whiteBalanceKelvin))K", highlight: engine.isWhiteBalanceLocked)
                                TelemetryPill(label: "FOCUS", value: "\(engine.focusMode.shortTitle) \(String(format: "%.2f", engine.lensPosition))", highlight: engine.focusMode == .manualFocus)
                                TelemetryPill(label: "STAB", value: engine.stabilizationMode.shortTitle, highlight: engine.stabilizationMode != .off)
                            }
                            .padding(.horizontal, 10)
                        }
                        .frame(width: screenW)
                        .padding(.top, 4)

                        // Row 2: Compact Action Pills + VU Meter + Histogram (Total width ~350pt < 390/428pt)
                        HStack(spacing: 5) {
                            Button {
                                engine.rotateOrientation90Degrees()
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: engine.orientationMode.isPortraitSwapped ? "rectangle.portrait" : "rectangle")
                                    Text(engine.orientationMode.shortLabel)
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 5)
                                .background(Color.black.opacity(0.68))
                                .foregroundColor(.yellow)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.yellow.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            Button {
                                showPostGradingSheet = true
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: "slider.horizontal.below.rectangle")
                                    Text("调色室")
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 5)
                                .background(Color.cyan.opacity(0.25))
                                .foregroundColor(.cyan)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            Button {
                                engine.performDeepMemoryAndStoragePurge()
                            } label: {
                                HStack(spacing: 3) {
                                    Image(systemName: "bolt.shield.fill")
                                    Text("清缓存")
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 5)
                                .background(Color.orange.opacity(0.25))
                                .foregroundColor(.orange)
                                .cornerRadius(6)
                                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.orange.opacity(0.5), lineWidth: 1))
                            }
                            .fixedSize()

                            Spacer(minLength: 2)

                            StereoAudioVUMeterView(leftDB: engine.audioPeakDBLeft, rightDB: engine.audioPeakDBRight)
                            LumaHistogramOverlay(bins: engine.lumaHistogram)
                        }
                        .padding(.horizontal, 10)
                        .frame(width: screenW)

                        // Row 3: Status Banner
                        HStack {
                            Text("\(engine.activeResolutionBadge) • \(engine.statusMessage)")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundColor(.white)
                                .lineLimit(1)
                                .truncationMode(.tail)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.black.opacity(0.62))
                                .cornerRadius(4)
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, 10)
                        .frame(width: screenW)
                    }

                    Spacer(minLength: 2)

                    // Centered Floating Lens Bar in Portrait
                    if !isLandscape {
                        HStack(spacing: 9) {
                            ForEach(CameraLensOption.allCases) { lens in
                                Button {
                                    engine.selectedLens = lens
                                } label: {
                                    Text(lens.rawValue)
                                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                                        .frame(width: 36, height: 36)
                                        .background(engine.selectedLens == lens ? Color.yellow : Color.black.opacity(0.68))
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
                                    .frame(height: 36)
                                    .background(engine.focusPeakingEnabled ? Color.green : Color.black.opacity(0.68))
                                    .foregroundColor(engine.focusPeakingEnabled ? .black : .white)
                                    .cornerRadius(18)
                                    .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color.white.opacity(0.25), lineWidth: 1))
                            }

                            Button {
                                engine.captureStillPhoto()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "camera.fill")
                                    Text("抓拍")
                                }
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 12)
                                .frame(height: 36)
                                .background(Color.white.opacity(0.92))
                                .foregroundColor(.black)
                                .cornerRadius(18)
                                .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color.yellow, lineWidth: 1))
                            }
                        }
                        .frame(width: screenW, alignment: .center)
                        .padding(.bottom, 2)
                    }

                    // Width-Clamped & Height-Capped Scrollable Control Drawer Card
                    if activeTab != .none {
                        VStack(spacing: 6) {
                            HStack(spacing: 6) {
                                Text(drawerTitle(for: activeTab))
                                    .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                    .foregroundColor(.yellow)
                                    .lineLimit(1)
                                Spacer(minLength: 4)
                                Button {
                                    withAnimation(.easeInOut(duration: 0.18)) {
                                        activeTab = .none
                                    }
                                } label: {
                                    HStack(spacing: 3) {
                                        Image(systemName: "xmark.circle.fill")
                                        Text("收起")
                                    }
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Color.white.opacity(0.16))
                                    .foregroundColor(.yellow)
                                    .cornerRadius(6)
                                }
                                .fixedSize()
                            }

                            ScrollView(.vertical, showsIndicators: true) {
                                VStack(spacing: 8) {
                                    if activeTab == .format {
                                        FormatAndStabilizationDrawer(engine: engine)
                                    } else if activeTab == .focus {
                                        FocusAndZoomDrawer(engine: engine)
                                    } else if activeTab == .exposure {
                                        ExposureAndWBDrawer(engine: engine, isLandscape: isLandscape)
                                    } else if activeTab == .lutGrading {
                                        LUTSystemDrawerView(engine: engine, showCubeImporter: $showCubeImporter, isLandscape: isLandscape)
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
                                .frame(maxWidth: .infinity)
                            }
                            .frame(maxHeight: isLandscape ? min(180, max(115, screenH - 145)) : min(270, screenH * 0.38))
                        }
                        .padding(10)
                        .frame(width: cardWidth)
                        .background(Color.black.opacity(0.90))
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.20), lineWidth: 1))
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                    }

                    // === SYMMETRICAL & WIDTH-LOCKED BOTTOM CONTROL DOCK ===
                    if isLandscape {
                        HStack(spacing: 8) {
                            HStack(spacing: 4) {
                                ForEach(CameraLensOption.allCases) { lens in
                                    Button {
                                        engine.selectedLens = lens
                                    } label: {
                                        Text(lens.rawValue)
                                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                                            .frame(width: 28, height: 28)
                                            .background(engine.selectedLens == lens ? Color.yellow : Color.white.opacity(0.14))
                                            .foregroundColor(engine.selectedLens == lens ? .black : .white)
                                            .clipShape(Circle())
                                    }
                                }
                                Button {
                                    engine.focusPeakingEnabled.toggle()
                                } label: {
                                    Text("PEAK")
                                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                                        .padding(.horizontal, 6)
                                        .frame(height: 28)
                                        .background(engine.focusPeakingEnabled ? Color.green : Color.white.opacity(0.14))
                                        .foregroundColor(engine.focusPeakingEnabled ? .black : .white)
                                        .cornerRadius(14)
                                }
                            }
                            .fixedSize()

                            HStack(spacing: 4) {
                                EqualWidthDrawerTabButton(title: "画幅", icon: "gyroscope", isActive: activeTab == .format) {
                                    toggleTab(.format)
                                }
                                EqualWidthDrawerTabButton(title: "对焦", icon: "viewfinder", isActive: activeTab == .focus) {
                                    toggleTab(.focus)
                                }
                                EqualWidthDrawerTabButton(title: "曝光", icon: "camera.aperture", isActive: activeTab == .exposure) {
                                    toggleTab(.exposure)
                                }
                                EqualWidthDrawerTabButton(title: "调色", icon: "cube", isActive: activeTab == .lutGrading) {
                                    toggleTab(.lutGrading)
                                }
                                EqualWidthDrawerTabButton(title: "监看", icon: "waveform", isActive: activeTab == .monitor) {
                                    toggleTab(.monitor)
                                }
                            }

                            VStack(alignment: .trailing, spacing: 2) {
                                HStack(spacing: 4) {
                                    ForEach(CaptureModeOption.allCases) { mode in
                                        Button {
                                            engine.captureMode = mode
                                        } label: {
                                            Text(mode == .video ? "VIDEO" : "PHOTO")
                                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                                .padding(.horizontal, 5)
                                                .padding(.vertical, 2)
                                                .background(engine.captureMode == mode ? Color.yellow.opacity(0.25) : Color.clear)
                                                .foregroundColor(engine.captureMode == mode ? .yellow : .gray)
                                                .cornerRadius(4)
                                        }
                                    }
                                }
                                Text("\(formatTimecode(engine.recordingDuration, fps: engine.selectedFPS.rawValue)) • \(engine.selectedCodec.rawValue.replacingOccurrences(of: "ProRes ", with: ""))")
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(engine.isRecording ? .red : .white)
                            }
                            .fixedSize()

                            Button {
                                engine.captureStillPhoto()
                            } label: {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 13, weight: .bold))
                                    .frame(width: 34, height: 34)
                                    .background(Color.white.opacity(0.90))
                                    .foregroundColor(.black)
                                    .clipShape(Circle())
                            }
                            .fixedSize()

                            Button(action: { engine.triggerMainShutter() }) {
                                ZStack {
                                    Circle()
                                        .stroke(Color.white, lineWidth: 2.5)
                                        .frame(width: 46, height: 46)
                                    if engine.captureMode == .photo {
                                        Circle()
                                            .fill(Color.white)
                                            .frame(width: 36, height: 36)
                                            .overlay(
                                                Image(systemName: "camera.aperture")
                                                    .font(.system(size: 16, weight: .bold))
                                                    .foregroundColor(.black)
                                            )
                                    } else if engine.isRecording {
                                        RoundedRectangle(cornerRadius: 6)
                                            .fill(Color.red)
                                            .frame(width: 20, height: 20)
                                    } else {
                                        Circle()
                                            .fill(Color.red)
                                            .frame(width: 36, height: 36)
                                    }
                                }
                                .contentShape(Circle())
                            }
                            .buttonStyle(.plain)
                            .fixedSize()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .frame(width: screenW)
                        .background(Color.black.opacity(0.86))
                        .layoutPriority(10)
                    } else {
                        // === PORTRAIT BOTTOM DOCK (100% Centered & Equal-Width 5-Tab Grid) ===
                        VStack(spacing: 7) {
                            // 5 Equal-Width Tabs (Never Clips or Overflows!)
                            HStack(spacing: 5) {
                                EqualWidthDrawerTabButton(title: "画幅/防抖", icon: "gyroscope", isActive: activeTab == .format) {
                                    toggleTab(.format)
                                }
                                EqualWidthDrawerTabButton(title: "对焦/跟焦", icon: "viewfinder", isActive: activeTab == .focus) {
                                    toggleTab(.focus)
                                }
                                EqualWidthDrawerTabButton(title: "曝光/色温", icon: "camera.aperture", isActive: activeTab == .exposure) {
                                    toggleTab(.exposure)
                                }
                                EqualWidthDrawerTabButton(title: "实时调色", icon: "cube", isActive: activeTab == .lutGrading) {
                                    toggleTab(.lutGrading)
                                }
                                EqualWidthDrawerTabButton(title: "监看/缓存", icon: "waveform", isActive: activeTab == .monitor) {
                                    toggleTab(.monitor)
                                }
                            }
                            .padding(.horizontal, 10)
                            .frame(width: screenW)

                            // Centered Mode Switcher + Compact Dual-Save Pill (No Bulky UISwitch!)
                            HStack(spacing: 10) {
                                ForEach(CaptureModeOption.allCases) { mode in
                                    Button {
                                        engine.captureMode = mode
                                    } label: {
                                        Text(mode.title)
                                            .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 4)
                                            .background(engine.captureMode == mode ? Color.yellow.opacity(0.24) : Color.white.opacity(0.06))
                                            .foregroundColor(engine.captureMode == mode ? .yellow : .gray)
                                            .cornerRadius(6)
                                    }
                                }

                                if engine.captureMode == .photo {
                                    Button {
                                        engine.dualSaveRawAndGradedPhoto.toggle()
                                    } label: {
                                        HStack(spacing: 3) {
                                            Image(systemName: engine.dualSaveRawAndGradedPhoto ? "checkmark.circle.fill" : "circle")
                                            Text("双存原片+LUT")
                                        }
                                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(engine.dualSaveRawAndGradedPhoto ? Color.cyan.opacity(0.25) : Color.white.opacity(0.08))
                                        .foregroundColor(engine.dualSaveRawAndGradedPhoto ? .cyan : .gray)
                                        .cornerRadius(6)
                                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(engine.dualSaveRawAndGradedPhoto ? Color.cyan.opacity(0.6) : Color.clear, lineWidth: 1))
                                    }
                                }
                            }
                            .frame(width: screenW, alignment: .center)

                            // Symmetrical 3-Column Shutter Bar (Left & Right have equal maxWidth: .infinity so Shutter is at exact 50% center!)
                            HStack(spacing: 0) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(formatTimecode(engine.recordingDuration, fps: engine.selectedFPS.rawValue))
                                        .font(.system(size: 15, weight: .bold, design: .monospaced))
                                        .foregroundColor(engine.isRecording ? .red : .white)
                                        .lineLimit(1)
                                    Text("\(engine.projectReelName)_T\(String(format: "%02d", engine.clipTakeNumber)) • \(engine.selectedCodec.rawValue)")
                                        .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                                        .foregroundColor(.yellow)
                                        .lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)

                                Button(action: { engine.triggerMainShutter() }) {
                                    ZStack {
                                        Circle()
                                            .stroke(Color.white, lineWidth: 3)
                                            .frame(width: 64, height: 64)
                                        if engine.captureMode == .photo {
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: 50, height: 50)
                                                .overlay(
                                                    Image(systemName: "camera.aperture")
                                                        .font(.system(size: 21, weight: .bold))
                                                        .foregroundColor(.black)
                                                )
                                        } else if engine.isRecording {
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(Color.red)
                                                .frame(width: 26, height: 26)
                                        } else {
                                            Circle()
                                                .fill(Color.red)
                                                .frame(width: 50, height: 50)
                                        }
                                    }
                                    .contentShape(Circle())
                                }
                                .buttonStyle(.plain)
                                .frame(width: 72)

                                VStack(alignment: .trailing, spacing: 2) {
                                    Text(engine.bakeIntoRecording ? "REC: LUT直出" : "REC: 纯净LOG")
                                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                                        .foregroundColor(engine.bakeIntoRecording ? .orange : .green)
                                        .lineLimit(1)
                                    Text("防抖:\(engine.stabilizationMode.shortTitle) • \(engine.orientationMode.shortLabel)")
                                        .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                                        .foregroundColor(.cyan)
                                        .lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            }
                            .padding(.horizontal, 14)
                            .padding(.bottom, 6)
                            .frame(width: screenW)
                        }
                        .padding(.top, 8)
                        .frame(width: screenW)
                        .background(Color.black.opacity(0.85))
                        .layoutPriority(10)
                    }
                }
                .frame(width: screenW, height: screenH)
            }
            .frame(width: screenW, height: screenH)
        }
        .preferredColorScheme(.dark)
        .onAppear {
            engine.requestPermissionsAndStart()
        }
        .fileImporter(
            isPresented: $showCubeImporter,
            allowedContentTypes: [UTType(filenameExtension: "cube") ?? .data, .plainText, .item],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let firstURL = urls.first {
                engine.importAndPersistResidentCubeLUT(from: firstURL)
            }
        }
        .sheet(isPresented: $showPostGradingSheet) {
            PostProductionGradingSheetView(engine: engine, postGrader: postGrader)
        }
        .sheet(isPresented: $showCubeShareSheet) {
            if let url = exportedCubeURL {
                ShareActivityView(activityItems: [url])
            }
        }
    }

    private func drawerTitle(for tab: ActiveControlDrawer) -> String {
        switch tab {
        case .none:       return ""
        case .format:     return "画幅 (4K / 1080p / Open Gate 4:3) & 防抖"
        case .focus:      return "专业对焦 (MF / A↔B 自动跟焦 / 峰值)"
        case .exposure:   return "手动曝光 (快门角度 / ISO) & 色温锁定"
        case .lutGrading: return "3D LUT 调色台 (支持永久驻留 .cube)"
        case .monitor:    return "监看辅助 • 场记板 • 深度释放缓存"
        }
    }

    private func toggleTab(_ tab: ActiveControlDrawer) {
        withAnimation(.easeInOut(duration: 0.18)) {
            activeTab = (activeTab == tab) ? .none : tab
        }
    }

    private func formatTimecode(_ seconds: TimeInterval, fps: Double) -> String {
        if engine.useTimeOfDayTimecode && !engine.isRecording {
            let comps = Calendar.current.dateComponents([.hour, .minute, .second], from: Date())
            return String(format: "%02d:%02d:%02d:00", comps.hour ?? 0, comps.minute ?? 0, comps.second ?? 0)
        }
        let totalSecs = Int(seconds)
        let mins = (totalSecs % 3600) / 60
        let secs = totalSecs % 60
        let frames = Int((seconds - Double(totalSecs)) * fps)
        return String(format: "%02d:%02d:%02d", mins, secs, max(0, frames))
    }
}

// MARK: - 10. Modular Control Drawers (Compact, Symmetrical & Never Overflowing)

struct FormatAndStabilizationDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Sensor Format", selection: $engine.selectedSensorPreset) {
                ForEach(SensorFormatPreset.allCases) { preset in
                    Text(preset.shortBadge).tag(preset)
                }
            }
            .pickerStyle(.segmented)
            .disabled(engine.isRecording)

            Picker("Stabilization", selection: $engine.stabilizationMode) {
                ForEach(CinemaStabilizationMode.allCases) { mode in
                    Text(mode.shortTitle).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 6) {
                Picker("FPS", selection: $engine.selectedFPS) {
                    ForEach(FrameRateOption.allCases) { fps in
                        Text(fps.label).tag(fps)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(engine.isRecording)

                Picker("Codec", selection: $engine.selectedCodec) {
                    ForEach(RecordingCodecOption.allCases) { codec in
                        Text(codec.rawValue.replacingOccurrences(of: "ProRes ", with: "")).tag(codec)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(engine.isRecording)
            }

            HStack(spacing: 10) {
                CompactTogglePill(title: "重力旋转", isOn: $engine.autoFollowDeviceOrientation, activeColor: .yellow)
                CompactTogglePill(title: "电子水平仪", isOn: $engine.showHorizonLeveler, activeColor: .green)
                CompactTogglePill(title: "水平镜像", isOn: $engine.isMirroredHorizontal, activeColor: .cyan)
                Spacer(minLength: 0)
            }
        }
    }
}

struct FocusAndZoomDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Picker("Focus Mode", selection: $engine.focusMode) {
                    ForEach(FocusControlMode.allCases) { mode in
                        Text(mode.shortTitle).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Picker("AF Range", selection: $engine.afRangeRestriction) {
                    ForEach(AutoFocusRangeOption.allCases) { range in
                        Text(range.title).tag(range)
                    }
                }
                .pickerStyle(.menu)

                CompactTogglePill(title: "平滑AF", isOn: $engine.isSmoothAFEnabled, activeColor: .yellow)
            }

            CompactSliderRow(
                title: "MF 对焦环",
                value: Binding(
                    get: { engine.lensPosition },
                    set: { engine.setManualFocusLensPosition($0) }
                ),
                range: 0.0...1.0,
                format: "%.2f"
            )

            HStack(spacing: 5) {
                Button("设A(\(String(format: "%.2f", engine.rackPointA)))") {
                    engine.rackPointA = engine.lensPosition
                }
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6).padding(.vertical, 4)
                .background(Color.white.opacity(0.14)).cornerRadius(5)

                Button("拉焦→A") {
                    engine.triggerRackFocus(to: engine.rackPointA)
                }
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6).padding(.vertical, 4)
                .background(Color.yellow.opacity(0.28)).foregroundColor(.yellow).cornerRadius(5)

                Button("设B(\(String(format: "%.2f", engine.rackPointB)))") {
                    engine.rackPointB = engine.lensPosition
                }
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6).padding(.vertical, 4)
                .background(Color.white.opacity(0.14)).cornerRadius(5)

                Button("拉焦→B") {
                    engine.triggerRackFocus(to: engine.rackPointB)
                }
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6).padding(.vertical, 4)
                .background(Color.yellow.opacity(0.28)).foregroundColor(.yellow).cornerRadius(5)

                Spacer(minLength: 2)

                Text(String(format: "%.1fs", engine.rackDurationSeconds))
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                Stepper("", value: $engine.rackDurationSeconds, in: 0.3...5.0, step: 0.2)
                    .labelsHidden()
                    .scaleEffect(0.85)
            }

            HStack(spacing: 8) {
                CompactTogglePill(title: "峰值对焦 Peaking", isOn: $engine.focusPeakingEnabled, activeColor: .green)
                Picker("Color", selection: $engine.focusPeakingColor) {
                    ForEach(FocusPeakingColorOption.allCases) { col in
                        Text(col.title).tag(col)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
    }
}

struct ExposureAndWBDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine
    var isLandscape: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                CompactTogglePill(title: "手动曝光 (快门+ISO)", isOn: $engine.isManualExposure, activeColor: .yellow)
                CompactTogglePill(title: "锁定白平衡 (K+Tint)", isOn: $engine.isWhiteBalanceLocked, activeColor: .cyan)
                Spacer(minLength: 0)
            }

            if isLandscape {
                HStack(spacing: 12) {
                    if engine.isManualExposure {
                        VStack(spacing: 4) {
                            CompactSliderRow(
                                title: "ISO 感光度",
                                value: Binding(get: { engine.currentISO }, set: { engine.setManualISO($0) }),
                                range: engine.minISO...engine.maxISO,
                                format: "%.0f"
                            )
                            CompactSliderRow(
                                title: "快门角度",
                                value: Binding(get: { Float(engine.shutterAngleDegrees) }, set: { engine.shutterAngleDegrees = Double($0) }),
                                range: 45.0...360.0,
                                format: "%.0f°"
                            )
                        }
                    }
                    if engine.isWhiteBalanceLocked {
                        VStack(spacing: 4) {
                            CompactSliderRow(title: "色温 Kelvin", value: $engine.whiteBalanceKelvin, range: 2500...9000, format: "%.0fK")
                            CompactSliderRow(title: "色调 Tint", value: $engine.whiteBalanceTint, range: -100...100, format: "%+.0f")
                        }
                    }
                }
            } else {
                if engine.isManualExposure {
                    CompactSliderRow(
                        title: "ISO 感光度",
                        value: Binding(get: { engine.currentISO }, set: { engine.setManualISO($0) }),
                        range: engine.minISO...engine.maxISO,
                        format: "%.0f"
                    )
                    CompactSliderRow(
                        title: "快门角度",
                        value: Binding(get: { Float(engine.shutterAngleDegrees) }, set: { engine.shutterAngleDegrees = Double($0) }),
                        range: 45.0...360.0,
                        format: "%.0f°"
                    )
                }

                if engine.isWhiteBalanceLocked {
                    CompactSliderRow(title: "色温 Kelvin", value: $engine.whiteBalanceKelvin, range: 2500...9000, format: "%.0fK")
                    CompactSliderRow(title: "色调 Tint", value: $engine.whiteBalanceTint, range: -100...100, format: "%+.0f")
                }
            }
        }
    }
}

struct LUTSystemDrawerView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @Binding var showCubeImporter: Bool
    var isLandscape: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Action Row: Import Resident LUT + Bake Toggle + Reset (Fits cleanly inside 370pt!)
            HStack(spacing: 6) {
                Button {
                    showCubeImporter = true
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "pin.fill")
                        Text("导入驻留 .cube")
                    }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.yellow.opacity(0.25))
                    .foregroundColor(.yellow)
                    .cornerRadius(6)
                }
                .fixedSize()

                CompactTogglePill(title: "烧录LUT入视频", isOn: $engine.bakeIntoRecording, activeColor: .orange)

                Spacer(minLength: 2)

                Button {
                    engine.lutSettings.reset()
                } label: {
                    Text("重置参数")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 4)
                        .background(Color.red.opacity(0.22))
                        .foregroundColor(.red)
                        .cornerRadius(6)
                }
                .fixedSize()
            }

            // Built-in LUT Presets
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(BuiltInLUTPreset.allCases) { preset in
                        Button {
                            engine.lutSettings.lutPreset = preset
                        } label: {
                            Text(preset.title)
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 9)
                                .padding(.vertical, 5)
                                .background(engine.lutSettings.lutPreset == preset ? Color.yellow : Color.white.opacity(0.12))
                                .foregroundColor(engine.lutSettings.lutPreset == preset ? .black : .white)
                                .cornerRadius(6)
                        }
                    }
                }
            }

            // Resident .cube LUT Library Row
            if !engine.residentLUTs.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        Text("📌 驻留库:")
                            .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                            .foregroundColor(.cyan)

                        ForEach(engine.residentLUTs) { item in
                            let isSelected = (engine.lutSettings.lutPreset == .customCubeLUT && engine.activeResidentLUTFileName == item.fileName)
                            HStack(spacing: 4) {
                                Button {
                                    engine.selectResidentLUT(item)
                                } label: {
                                    Text(item.displayName)
                                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                                        .foregroundColor(isSelected ? .black : .cyan)
                                }
                                Button {
                                    engine.deleteResidentLUT(item)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.system(size: 10))
                                        .foregroundColor(isSelected ? .black.opacity(0.7) : .gray)
                                }
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(isSelected ? Color.cyan : Color.cyan.opacity(0.15))
                            .cornerRadius(6)
                        }
                    }
                }
            }

            if engine.lutSettings.lutPreset == .customCubeLUT {
                Picker("LUT Input Space", selection: $engine.lutSettings.lutInputSpace) {
                    ForEach(LUTInputSpaceOption.allCases) { opt in
                        Text(opt.title).tag(opt)
                    }
                }
                .pickerStyle(.segmented)
            }

            if isLandscape {
                HStack(alignment: .top, spacing: 12) {
                    VStack(spacing: 4) {
                        CompactSliderRow(title: "LUT 强度", value: $engine.lutSettings.lutIntensity, range: 0.0...1.0, format: "%.2f")
                        CompactSliderRow(title: "曝光 EV", value: $engine.lutSettings.exposureEV, range: -2.0...2.0, format: "%+.2f")
                        CompactSliderRow(title: "对比度", value: $engine.lutSettings.contrast, range: 0.6...1.5, format: "%.2f")
                    }
                    VStack(spacing: 4) {
                        if engine.selectedSensorPreset == .openGateMax {
                            CompactSliderRow(title: "去断层", value: $engine.lutSettings.debanding, range: 0.0...1.0, format: "%.2f")
                        }
                        CompactSliderRow(title: "高光柔和", value: $engine.lutSettings.highlightRollOff, range: 0.0...1.0, format: "%.2f")
                        CompactSliderRow(title: "饱和度", value: $engine.lutSettings.saturation, range: 0.0...1.8, format: "%.2f")
                    }
                }
            } else {
                VStack(spacing: 4) {
                    CompactSliderRow(title: "LUT 强度", value: $engine.lutSettings.lutIntensity, range: 0.0...1.0, format: "%.2f")
                    if engine.selectedSensorPreset == .openGateMax {
                        CompactSliderRow(title: "OpenGate去断层", value: $engine.lutSettings.debanding, range: 0.0...1.0, format: "%.2f")
                    }
                    CompactSliderRow(title: "曝光补偿 EV", value: $engine.lutSettings.exposureEV, range: -2.0...2.0, format: "%+.2f")
                    CompactSliderRow(title: "对比度", value: $engine.lutSettings.contrast, range: 0.6...1.5, format: "%.2f")
                    CompactSliderRow(title: "高光柔和 Roll", value: $engine.lutSettings.highlightRollOff, range: 0.0...1.0, format: "%.2f")
                    CompactSliderRow(title: "饱和度 Sat", value: $engine.lutSettings.saturation, range: 0.0...1.8, format: "%.2f")
                }
            }
        }
    }
}

struct MonitorAndSlateDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine
    let onExportCube: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Diagnostic", selection: $engine.monitorDiagnostic) {
                ForEach(MonitorDiagnosticMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Picker("Aspect Guide", selection: $engine.aspectGuide) {
                ForEach(AspectGuideOption.allCases) { guide in
                    Text(guide.rawValue).tag(guide)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 8) {
                HStack(spacing: 4) {
                    Text("卷号:")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    TextField("A001", text: $engine.projectReelName)
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 56)
                }
                Stepper("镜号 T\(String(format: "%02d", engine.clipTakeNumber))", value: $engine.clipTakeNumber, in: 1...99)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))

                Spacer(minLength: 2)

                Button("导出 .cube") {
                    onExportCube()
                }
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.green.opacity(0.25))
                .foregroundColor(.green)
                .cornerRadius(6)
                .fixedSize()
            }

            VStack(alignment: .leading, spacing: 6) {
                Text(String(format: "显存/RAM: %.0fMB • 沙盒缓存: %.1fMB • 可用空间: %.1fGB",
                            engine.memoryFootprintMB, engine.tempStorageUsedMB, engine.deviceFreeSpaceGB))
                    .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)
                    .lineLimit(1)

                HStack(spacing: 6) {
                    Button {
                        engine.performDeepMemoryAndStoragePurge()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "bolt.shield.fill")
                            Text("深度释放内存与缓存")
                        }
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.orange.opacity(0.28))
                        .foregroundColor(.orange)
                        .cornerRadius(6)
                    }

                    if !engine.sessionSavedAssetIDs.isEmpty {
                        Button {
                            engine.deleteSessionVideosFromPhotoLibrary()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "trash.fill")
                                Text("删测试视频(\(engine.sessionSavedAssetIDs.count))")
                            }
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 9)
                            .padding(.vertical, 5)
                            .background(Color.red.opacity(0.28))
                            .foregroundColor(.red)
                            .cornerRadius(6)
                        }
                    }
                    Spacer(minLength: 0)
                }
            }

            HStack(spacing: 8) {
                CompactTogglePill(title: "录制存相册", isOn: $engine.autoSaveVideoToPhotos, activeColor: .yellow)
                CompactTogglePill(title: "九宫格线", isOn: $engine.showGridOverlay, activeColor: .green)
                CompactTogglePill(title: "真实时间码", isOn: $engine.useTimeOfDayTimecode, activeColor: .cyan)
                Spacer(minLength: 0)
            }
        }
        .onAppear {
            engine.refreshTempStorageUsage()
        }
    }
}

// MARK: - 11. Post-Production Apple Log Grading Sheet View

struct PostProductionGradingSheetView: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @ObservedObject var postGrader: PostProductionLogGrader
    @Environment(\.dismiss) private var dismiss
    @State private var showVideoPicker: Bool = false

    var body: some View {
        NavigationView {
            VStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black)
                        .frame(height: 230)

                    if let img = postGrader.gradedPreviewImage {
                        Image(uiImage: img)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 220)
                            .cornerRadius(8)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "film.stack")
                                .font(.system(size: 34))
                                .foregroundColor(.gray)
                            Text("加载最新录制内存帧或从文件导入 Apple Log 视频调色")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.gray)
                        }
                    }
                }
                .padding(.horizontal)

                HStack(spacing: 12) {
                    if engine.lastRecordedPreviewFrame != nil {
                        Button("加载最新录制帧") {
                            postGrader.loadCachedPreviewFrame(from: engine)
                        }
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.yellow)
                        .foregroundColor(.black)
                        .cornerRadius(6)
                    }

                    Button("从文件选择 .mov") {
                        showVideoPicker = true
                    }
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .cornerRadius(6)

                    if postGrader.gradedPreviewImage != nil {
                        Button("保存调色帧至相册") {
                            postGrader.saveGradedPhotoToLibrary()
                        }
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.green)
                        .foregroundColor(.black)
                        .cornerRadius(6)
                    }
                }

                if !postGrader.exportStatus.isEmpty {
                    Text(postGrader.exportStatus)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(.green)
                }

                ScrollView {
                    VStack(spacing: 10) {
                        LUTSystemDrawerView(engine: engine, showCubeImporter: .constant(false), isLandscape: false)
                    }
                    .padding()
                }
                .onChange(of: engine.lutSettings) { _ in
                    postGrader.renderGradedStill(engine: engine)
                }
                .onAppear {
                    if postGrader.sourceImage == nil {
                        postGrader.loadCachedPreviewFrame(from: engine)
                    }
                }
            }
            .navigationTitle("Apple Log 后期无损调色室")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
            .fileImporter(
                isPresented: $showVideoPicker,
                allowedContentTypes: [.movie, .quickTimeMovie, .mpeg4Movie],
                allowsMultipleSelection: false
            ) { res in
                if case .success(let urls) = res, let url = urls.first {
                    let accessed = url.startAccessingSecurityScopedResource()
                    postGrader.extractPreviewFrame(from: url, engine: engine)
                    if accessed { url.stopAccessingSecurityScopedResource() }
                }
            }
        }
        .preferredColorScheme(.dark)
    }
}

// MARK: - 12. Reusable UI Components (Equal-Width Tabs, Capsule Toggles & Fitted Sliders)

struct TelemetryPill: View {
    let label: String
    let value: String
    let highlight: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.system(size: 7.5, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                .foregroundColor(highlight ? .yellow : .white)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Color.black.opacity(0.68))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(highlight ? Color.yellow.opacity(0.45) : Color.white.opacity(0.15), lineWidth: 1)
        )
        .fixedSize()
    }
}

struct EqualWidthDrawerTabButton: View {
    let title: String
    let icon: String
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 3) {
                Image(systemName: icon)
                    .font(.system(size: 9.5, weight: .bold))
                Text(title)
                    .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .background(isActive ? Color.yellow : Color.white.opacity(0.12))
            .foregroundColor(isActive ? .black : .white)
            .cornerRadius(7)
        }
    }
}

struct CompactTogglePill: View {
    let title: String
    @Binding var isOn: Bool
    let activeColor: Color

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 10, weight: .bold))
                Text(title)
                    .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                    .lineLimit(1)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(isOn ? activeColor.opacity(0.24) : Color.white.opacity(0.08))
            .foregroundColor(isOn ? activeColor : .gray)
            .cornerRadius(6)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isOn ? activeColor.opacity(0.55) : Color.white.opacity(0.12), lineWidth: 1)
            )
        }
        .fixedSize()
    }
}

struct CompactSliderRow: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                .foregroundColor(.gray)
                .frame(width: 78, alignment: .leading)
                .lineLimit(1)
            Slider(value: $value, in: range)
                .accentColor(.yellow)
            Text(String(format: format, value))
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
                .frame(width: 40, alignment: .trailing)
                .lineLimit(1)
        }
        .padding(.horizontal, 2)
    }
}

struct ShareActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - 13. Application Entry Point

@main
struct CineLog13ProMaxApp: App {
    var body: some Scene {
        WindowGroup {
            CineLogMainCameraView()
        }
    }
}