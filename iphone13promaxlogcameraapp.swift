//
//  iPhone13ProMaxLogCameraApp.swift
//  CineLog13Pro - 10-Bit Apple Log, Open Gate & Live/Post Color Grading Camera for iPhone 13 Pro Max
//
//  Target: iOS 16.0+ (Optimized for iPhone 13 Pro Max A15 Bionic & compatible with iPhone 15 Pro+)
//  Frameworks: UIKit, SwiftUI, AVFoundation, AVKit, Metal, MetalKit, CoreVideo, CoreMedia, VideoToolbox, Photos, PhotosUI, UniformTypeIdentifiers
//

import UIKit
import SwiftUI
import AVFoundation
import AVKit
import Metal
import MetalKit
import CoreVideo
import CoreMedia
import VideoToolbox
import Photos
import PhotosUI
import QuartzCore
import UniformTypeIdentifiers
import Combine

// MARK: - 1. Embedded Metal Compute & Render Shaders (Orientation + Open Gate + Apple Log + Color Grading + 3D LUT)

enum MetalShaderLibrarySource {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct ColorGradeUniforms {
        int isNativeAppleLog;       // 1 = hardware native Apple Log, 0 = HLG/SDR sensor input
        int isFullRangeYCbCr;       // 1 = Full Range ('420f'), 0 = Video Range ('x422', 'x420', '420v')
        int isSDRSensorInput;       // 1 = 8-bit SDR P3/709 sensor fallback (some Open Gate modes), 0 = 10-bit HLG BT.2020
        int rotationMode;           // 0 = Portrait (90 CW), 1 = Landscape Right (0 deg), 2 = Landscape Left (180 deg), 3 = Portrait UpsideDown (270 CW)
        int mirrorHorizontal;       // 1 = Mirror X, 0 = Normal

        int lookPreset;             // 0 = Raw Log, 1 = Official Rec.709, 2 = Kodak 2383, 3 = Fuji Eterna, 4 = Teal & Orange, 5 = Bleach Bypass, 6 = Custom 3D LUT
        int bakeGradeIntoRecording; // 1 = Bake grade into recorded file, 0 = Record pure Apple Log & only grade viewfinder
        int monitorDiagnosticMode;  // 0 = Normal, 1 = False Color IRE, 2 = Zebra (>94% IRE)
        int hasCustom3DLUT;         // 1 = Custom .cube 3D texture bound at texture(4)

        float exposureEV;           // -3.0 to +3.0 EV
        float contrast;             // 0.5 to 1.8 (1.0 = neutral)
        float pivot;                // Contrast pivot in Log space (default 0.488 = 18% gray)
        float highlightRollOff;     // 0.0 to 1.0 shoulder compression
        float shadowLift;           // -0.25 to +0.25 shadow toe adjustment
        float saturation;           // 0.0 to 2.0
        float vibrance;             // -1.0 to +1.0
        float temperatureOffset;    // -1.0 (Cool) to +1.0 (Warm)
        float tintOffset;           // -1.0 (Green) to +1.0 (Magenta)
        float lutIntensity;         // 0.0 to 1.0 blend for Look / 3D LUT
        float filmGrain;            // 0.0 to 1.0 organic 35mm grain
        float frameTime;            // Time seed for grain & zebra animation

        float4 liftRGB;             // Shadow RGB offset (-0.2 to +0.2)
        float4 gammaRGB;            // Midtone RGB power/offset (-0.3 to +0.3)
        float4 gainRGB;             // Highlight RGB multiplier (0.5 to 1.5, default 1.0)
    };

    // --- 1. YCbCr to Normalized RGB Conversion (Supports 10-bit Video Range & 8-bit Full/Video Range) ---
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
            // ITU-R BT.709 / Display P3 YCbCr matrix
            float r = y + 1.5748 * cr;
            float g = y - 0.187324 * cb - 0.468124 * cr;
            float b = y + 1.8556 * cb;
            return clamp(float3(r, g, b), 0.0, 1.0);
        } else {
            // ITU-R BT.2020 Non-Constant Luminance matrix
            float r = y + 1.4746 * cr;
            float g = y - 0.16455312684366 * cb - 0.57135312684366 * cr;
            float b = y + 1.8814 * cb;
            return clamp(float3(r, g, b), 0.0, 1.0);
        }
    }

    // --- 2. ITU-R BT.2100 HLG & SDR Inverse OETF to Scene-Linear BT.2020 ---
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
            // Convert P3/sRGB gamma ~2.2 to Linear and map into BT.2020 primaries
            float3 linP3 = pow(max(encodedRGB, 0.0), float3(2.2));
            sceneLinear.r = 0.753833 * linP3.r + 0.198597 * linP3.g + 0.047570 * linP3.b;
            sceneLinear.g = 0.045744 * linP3.r + 0.941777 * linP3.g + 0.012479 * linP3.b;
            sceneLinear.b = -0.001210 * linP3.r + 0.017602 * linP3.g + 0.983608 * linP3.b;
            sceneLinear = max(sceneLinear, 0.0) * exp2(evBias);
        } else {
            sceneLinear = float3(
                hlgInverseOETFChannel(encodedRGB.r),
                hlgInverseOETFChannel(encodedRGB.g),
                hlgInverseOETFChannel(encodedRGB.b)
            );
            sceneLinear *= exp2(evBias) * 2.4;
        }

        if (rollOff > 0.001) {
            float3 shoulder = sceneLinear / (1.0 + sceneLinear * (0.28 * rollOff));
            sceneLinear = mix(sceneLinear, shoulder, rollOff);
        }
        return sceneLinear;
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

    // --- 4. Real-Time & Post Color Grading Engine (Primary + 3-Way LGG + Creative Film Looks + 3D LUT) ---
    inline float3 applyColorGradingPipeline(
        float3 appleLogIn,
        constant ColorGradeUniforms& u,
        texture3d<float, access::sample> customLUT,
        uint2 pixelCoord
    ) {
        float3 logColor = appleLogIn;

        // (A) White Balance Offset (Temperature & Tint in Log Space)
        logColor.r += u.temperatureOffset * 0.06 + u.tintOffset * 0.03;
        logColor.g -= u.tintOffset * 0.05;
        logColor.b -= u.temperatureOffset * 0.06 - u.tintOffset * 0.03;

        // (B) 3-Way Lift / Gamma / Gain in Log Space
        float3 lifted = logColor + u.liftRGB.xyz * (1.0 - logColor);
        float3 gained = lifted * u.gainRGB.xyz;
        float3 gammaPower = max(float3(0.2), 1.0 - u.gammaRGB.xyz * 0.65);
        logColor = pow(clamp(gained, 0.0, 1.0), gammaPower);

        // (C) Shadow Lift & Contrast around 18% Middle Gray Pivot (0.488)
        if (abs(u.shadowLift) > 0.001) {
            float shadowWeight = clamp(1.0 - (logColor.g / 0.5), 0.0, 1.0);
            logColor += u.shadowLift * shadowWeight * shadowWeight * 0.25;
        }
        logColor = (logColor - u.pivot) * u.contrast + u.pivot;
        logColor = clamp(logColor, 0.0, 1.0);

        // If user selected Raw Log (Preset 0), apply saturation in Log space and return
        if (u.lookPreset == 0) {
            float luma = dot(logColor, float3(0.2627, 0.6780, 0.0593));
            return clamp(mix(float3(luma), logColor, u.saturation), 0.0, 1.0);
        }

        // Convert Graded Apple Log -> Linear BT.2020 -> Linear Rec.709
        float3 lin2020 = appleLogToLinear(logColor);
        float3 lin709;
        lin709.r =  1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b;
        lin709.g = -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b;
        lin709.b = -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b;
        lin709 = max(lin709, 0.0);

        // Base ACES Filmic Tone Mapping to Rec.709 [0, 1]
        float3 x = lin709 * 1.38;
        float3 rec709 = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
        rec709 = pow(rec709, float3(1.0 / 2.2));

        float3 styled = rec709;
        float luma709 = dot(rec709, float3(0.2126, 0.7152, 0.0722));

        if (u.lookPreset == 2) {
            // Kodak 2383 Print Film Emulation: Rich toe contrast, warm golden highlights, cyan-blue split shadows
            float3 sCurve = smoothstep(0.02, 0.98, rec709);
            float3 shadowTint = float3(-0.03, 0.01, 0.045) * (1.0 - luma709);
            float3 hiTint     = float3(0.04, 0.025, -0.03) * luma709;
            styled = clamp(mix(rec709, sCurve, 0.45) + shadowTint + hiTint, 0.0, 1.0);
        } else if (u.lookPreset == 3) {
            // Fuji Eterna 250D: Soft highlight roll-off, lifted shadows, pastel cyan/emerald foliage
            float3 soft = mix(rec709, sqrt(rec709), 0.18);
            soft = soft * 0.90 + 0.04;
            soft.r *= 0.97;
            soft.g *= 1.02;
            soft.b *= 1.03;
            styled = clamp(soft, 0.0, 1.0);
        } else if (u.lookPreset == 4) {
            // Teal & Orange Blockbuster Look
            float3 shadows = float3(-0.06, 0.02, 0.08) * pow(1.0 - luma709, 1.4);
            float3 midsHi  = float3(0.07, 0.02, -0.05) * pow(luma709, 0.9);
            styled = clamp(smoothstep(0.0, 1.0, rec709) * 0.35 + rec709 * 0.65 + shadows + midsHi, 0.0, 1.0);
        } else if (u.lookPreset == 5) {
            // Bleach Bypass (Silver Retention): High metallic contrast + reduced saturation
            float3 desat = mix(float3(luma709), rec709, 0.42);
            float3 hard  = smoothstep(0.06, 0.94, desat);
            styled = clamp(mix(desat, hard, 0.6), 0.0, 1.0);
        } else if (u.lookPreset == 6 && u.hasCustom3DLUT == 1) {
            // Custom .cube 3D LUT Trilinear Sampling
            constexpr sampler lutSampler(coord::normalized, address::clamp_to_edge, filter::linear);
            styled = customLUT.sample(lutSampler, logColor).rgb;
        }

        // Blend by LUT Intensity
        float3 result = mix(rec709, styled, clamp(u.lutIntensity, 0.0, 1.0));

        // Saturation & Vibrance (Skin-tone aware saturation boost)
        float finalLuma = dot(result, float3(0.2126, 0.7152, 0.0722));
        float maxC = max(result.r, max(result.g, result.b));
        float minC = min(result.r, min(result.g, result.b));
        float currentSat = maxC - minC;
        float vibranceFactor = 1.0 + u.vibrance * (1.0 - currentSat);
        float totalSat = max(0.0, u.saturation * vibranceFactor);
        result = clamp(mix(float3(finalLuma), result, totalSat), 0.0, 1.0);

        // Optional 35mm Organic Film Grain
        if (u.filmGrain > 0.001) {
            float2 seed = float2(pixelCoord) * 0.137 + float2(u.frameTime * 11.3, u.frameTime * 7.9);
            float noise = fract(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453) - 0.5;
            float midMask = 4.0 * finalLuma * (1.0 - finalLuma);
            result = clamp(result + noise * u.filmGrain * 0.09 * midMask, 0.0, 1.0);
        }

        return result;
    }

    // --- 5. False Color IRE Exposure Diagnostic Map ---
    inline float3 falseColorMap(float lumaIRE) {
        if (lumaIRE < 0.05) return float3(0.55, 0.0, 0.85);       // Purple: Crushed Black (<5%)
        else if (lumaIRE < 0.18) return float3(0.1, 0.35, 0.95);  // Blue: Deep Shadows
        else if (lumaIRE < 0.44) return float3(0.2, 0.75, 0.85);  // Cyan: Lower Midtones
        else if (lumaIRE < 0.53) return float3(0.15, 0.85, 0.25); // Green: 18% Middle Gray Zone (0.488 in Apple Log)
        else if (lumaIRE < 0.62) return float3(0.95, 0.55, 0.65); // Pink: +1 EV Skin Tone Zone
        else if (lumaIRE < 0.82) return float3(0.8, 0.8, 0.8);    // Neutral Gray: Upper Highlights
        else if (lumaIRE < 0.94) return float3(0.98, 0.85, 0.1);  // Yellow: Near Clipping Warning
        else return float3(0.98, 0.1, 0.1);                       // Red: Clipped Highlight (>94%)
    }

    // --- 6. Main Camera Compute Kernel (Handles Orientation Rotation + Open Gate + Log + Live Grading) ---
    kernel void processAppleLogFrame(
        texture2d<float, access::read>   lumaPlane     [[texture(0)]],
        texture2d<float, access::read>   chromaPlane   [[texture(1)]],
        texture2d<float, access::write>  recordOutTex  [[texture(2)]],
        texture2d<float, access::write>  previewOutTex [[texture(3)]],
        texture3d<float, access::sample> customLUTTex  [[texture(4)]],
        constant ColorGradeUniforms&     uniforms      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint outW = recordOutTex.get_width();
        uint outH = recordOutTex.get_height();
        if (gid.x >= outW || gid.y >= outH) {
            return;
        }

        uint srcW = lumaPlane.get_width();
        uint srcH = lumaPlane.get_height();

        // Apply optional horizontal mirror first
        uint destX = (uniforms.mirrorHorizontal == 1) ? (outW - 1 - gid.x) : gid.x;
        uint destY = gid.y;

        // Map upright output pixel (destX, destY) back to physical sensor coordinates (srcX, srcY)
        uint srcX = 0;
        uint srcY = 0;
        if (uniforms.rotationMode == 0) {
            // Portrait (90° CW rotation from native LandscapeRight sensor)
            srcX = min(destY, srcW - 1);
            srcY = (srcH - 1) - min(destX, srcH - 1);
        } else if (uniforms.rotationMode == 1) {
            // Landscape Right / Standard Hold (0° native sensor orientation)
            srcX = min(destX, srcW - 1);
            srcY = min(destY, srcH - 1);
        } else if (uniforms.rotationMode == 2) {
            // Landscape Left (180° rotation)
            srcX = (srcW - 1) - min(destX, srcW - 1);
            srcY = (srcH - 1) - min(destY, srcH - 1);
        } else {
            // Portrait Upside Down (270° CW / 90° CCW)
            srcX = (srcW - 1) - min(destY, srcW - 1);
            srcY = min(destX, srcH - 1);
        }

        uint2 lumaCoord = uint2(srcX, srcY);
        uint2 chromaCoord = uint2(
            srcX * chromaPlane.get_width() / srcW,
            srcY * chromaPlane.get_height() / srcH
        );

        float  ySample    = lumaPlane.read(lumaCoord).r;
        float2 cbcrSample = chromaPlane.read(chromaCoord).rg;

        float3 sensorRGB = decodeYCbCrToRGB(ySample, cbcrSample, uniforms.isFullRangeYCbCr, uniforms.isSDRSensorInput);

        // Produce canonical Apple Log (BT.2020) signal
        float3 appleLogRGB;
        if (uniforms.isNativeAppleLog == 1) {
            appleLogRGB = sensorRGB;
        } else {
            float3 linearBT2020 = sensorToLinearBT2020(sensorRGB, uniforms.isSDRSensorInput, uniforms.exposureEV, uniforms.highlightRollOff);
            appleLogRGB = linearToAppleLog(linearBT2020);
        }

        // Compute Graded Color for Viewfinder (and optionally for Recorded File)
        float3 gradedRGB = applyColorGradingPipeline(appleLogRGB, uniforms, customLUTTex, gid);

        // Write Recording Buffer (either Pure Apple Log or Baked Color Grade)
        float3 recordSignal = (uniforms.bakeGradeIntoRecording == 1) ? gradedRGB : appleLogRGB;
        recordOutTex.write(float4(recordSignal, 1.0), gid);

        // Write Viewfinder Preview Buffer (with optional False Color or Zebra diagnostic overlay)
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

        previewOutTex.write(float4(monitorRGB, 1.0), gid);
    }

    // --- 7. Post-Production Video Frame Grading Kernel (Grades RGBA Texture from Imported Video) ---
    kernel void gradeExistingLogFrame(
        texture2d<float, access::read>   sourceTex    [[texture(0)]],
        texture2d<float, access::write>  gradedOutTex [[texture(1)]],
        texture3d<float, access::sample> customLUTTex [[texture(2)]],
        constant ColorGradeUniforms&     uniforms     [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= gradedOutTex.get_width() || gid.y >= gradedOutTex.get_height()) {
            return;
        }
        float3 srcLog = sourceTex.read(gid).rgb;
        if (abs(uniforms.exposureEV) > 0.001 || uniforms.highlightRollOff > 0.001) {
            float3 lin = appleLogToLinear(srcLog) * exp2(uniforms.exposureEV);
            if (uniforms.highlightRollOff > 0.001) {
                float3 shoulder = lin / (1.0 + lin * (0.28 * uniforms.highlightRollOff));
                lin = mix(lin, shoulder, uniforms.highlightRollOff);
            }
            srcLog = linearToAppleLog(lin);
        }
        float3 graded = applyColorGradingPipeline(srcLog, uniforms, customLUTTex, gid);
        gradedOutTex.write(float4(graded, 1.0), gid);
    }

    // --- 8. Render Pipeline Shaders for MTKView (Aspect-Fit Scaling to Any Screen Orientation) ---
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

// MARK: - 2. Camera & Color Grading Data Models

enum CaptureOrientationMode: Int, CaseIterable, Identifiable {
    case portrait           = 0 // 竖屏 (9:16 / 3:4)
    case landscapeRight     = 1 // 横屏-右 (16:9 / 4:3 标准握持)
    case landscapeLeft      = 2 // 横屏-左 (16:9 / 4:3 反向横屏)
    case portraitUpsideDown = 3 // 竖屏-倒置

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .portrait:           return "竖屏 (Portrait)"
        case .landscapeRight:     return "横屏 (Landscape)"
        case .landscapeLeft:      return "横屏反向 (Flip)"
        case .portraitUpsideDown: return "竖屏倒置 (270°)"
        }
    }

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

enum SensorFormatPreset: String, CaseIterable, Identifiable {
    case openGateMax = "Open Gate 4:3 (满传感器最大分辨率)"
    case uhd4K16x9   = "4K UHD 16:9 (3840×2160 10-Bit)"
    case fhd1080p    = "1080p FHD 16:9 (1920×1080 高帧率)"

    var id: String { rawValue }

    var shortTitle: String {
        switch self {
        case .openGateMax: return "Open Gate 4:3"
        case .uhd4K16x9:   return "4K 16:9"
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
        case .ultraWide: return "0.5x (13mm)"
        case .wideMain:  return "1x (26mm f/1.5)"
        case .telephoto: return "3x (77mm)"
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

enum LookPresetOption: Int, CaseIterable, Identifiable {
    case rawLog       = 0
    case rec709       = 1
    case kodak2383    = 2
    case fujiEterna   = 3
    case tealOrange   = 4
    case bleachBypass = 5
    case customCube   = 6

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .rawLog:       return "Raw Log"
        case .rec709:       return "Rec.709"
        case .kodak2383:    return "Kodak 2383"
        case .fujiEterna:   return "Fuji Eterna"
        case .tealOrange:   return "Teal & Orange"
        case .bleachBypass: return "Bleach Bypass"
        case .customCube:   return "Custom .cube"
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
    var lookPreset: LookPresetOption = .rec709
    var bakeGradeIntoRecording: Bool = false
    var lutIntensity: Float = 1.0

    var exposureEV: Float = 0.0
    var contrast: Float = 1.0
    var pivot: Float = 0.488
    var highlightRollOff: Float = 0.35
    var shadowLift: Float = 0.0
    var saturation: Float = 1.0
    var vibrance: Float = 0.0
    var temperatureOffset: Float = 0.0
    var tintOffset: Float = 0.0
    var filmGrain: Float = 0.0

    // 3-Way Color Wheels (Lift / Gamma / Gain RGB)
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
        let keepLook = lookPreset
        let keepBake = bakeGradeIntoRecording
        self = ColorGradeParameters()
        self.lookPreset = keepLook
        self.bakeGradeIntoRecording = keepBake
    }
}

// MARK: - 3. 3D .cube LUT Parser & Exporter

enum CubeLUTManager {
    static func loadCubeLUT(from url: URL, device: MTLDevice) throws -> (MTLTexture, String) {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }

        let content = try String(contentsOf: url, encoding: .utf8)
        var size = 0
        var floatValues: [Float] = []
        floatValues.reserveCapacity(33 * 33 * 33 * 4)

        for rawLine in content.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") || line.hasPrefix("TITLE") || line.hasPrefix("DOMAIN_") {
                continue
            }
            if line.hasPrefix("LUT_3D_SIZE") {
                let parts = line.split(separator: " ")
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
            throw NSError(domain: "CubeLUT", code: -1, userInfo: [NSLocalizedDescriptionKey: "无效的 .cube 3D LUT 文件格式（需包含 LUT_3D_SIZE）"])
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

        return (texture, url.deletingPathExtension().lastPathComponent)
    }

    static func exportCubeFile(grade: ColorGradeParameters) throws -> URL {
        let size = 33
        var lines: [String] = [
            "# Generated by CineLog 13 Pro Max (Apple Log Color Grading Engine)",
            "TITLE \"CineLog13Pro_\(grade.lookPreset.title)\"",
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

        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("CineLog13Pro_Grade.cube")
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

        let linR709: Float = max(0.0,  1.6604910 * linR2020 - 0.5876411 * linG2020 - 0.0728499 * linB2020) * 1.38
        let linG709: Float = max(0.0, -0.1245505 * linR2020 + 1.1328999 * linG2020 - 0.0083494 * linB2020) * 1.38
        let linB709: Float = max(0.0, -0.0181508 * linR2020 - 0.1005789 * linG2020 + 1.1187297 * linB2020) * 1.38

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

// MARK: - 4. Core 10-Bit Apple Log + Open Gate + Orientation + Live Color Grading Engine

final class AppleLogCameraEngine: NSObject, ObservableObject, @unchecked Sendable {
    @Published var isSessionRunning: Bool = false
    @Published var isRecording: Bool = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var activePipelineBadge: String = "Initializing 10-Bit Open Gate Pipeline..."
    @Published var activeResolutionBadge: String = "3840×2160"
    @Published var statusMessage: String = "Ready"
    @Published var lastRecordedVideoURL: URL?

    // Capture Format, Open Gate, Lens & Orientation Controls
    @Published var selectedSensorPreset: SensorFormatPreset = .openGateMax {
        didSet { reconfigureFormatAndSession() }
    }
    @Published var selectedLens: CameraLensOption = .wideMain {
        didSet { switchCameraLens(selectedLens) }
    }
    @Published var orientationMode: CaptureOrientationMode = .portrait
    @Published var autoFollowDeviceOrientation: Bool = false
    @Published var isMirroredHorizontal: Bool = false
    @Published var selectedCodec: RecordingCodecOption = .proRes422
    @Published var selectedFPS: FrameRateOption = .fps24 {
        didSet { reconfigureFormatAndSession() }
    }

    // Color Grading & Monitoring Controls
    @Published var grade = ColorGradeParameters()
    @Published var monitorDiagnostic: MonitorDiagnosticMode = .off
    @Published var aspectGuide: AspectGuideOption = .none
    @Published var showGridOverlay: Bool = true
    @Published var customLUTName: String = "未加载 .cube"

    // Manual Cinema Exposure & White Balance Controls
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
        didSet { applyWhiteBalanceKelvin(whiteBalanceKelvin) }
    }
    @Published var isWhiteBalanceLocked: Bool = false {
        didSet { toggleWhiteBalanceLock(isWhiteBalanceLocked) }
    }

    // Live Luma Histogram (64 bins)
    @Published var lumaHistogram: [Float] = Array(repeating: 0, count: 64)

    // AVFoundation Capture Stack
    let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.cinelog13pro.sessionQueue", qos: .userInteractive)
    private let videoProcessingQueue = DispatchQueue(label: "com.cinelog13pro.videoProcessingQueue", qos: .userInteractive)
    private var videoDevice: AVCaptureDevice?
    private var currentVideoInput: AVCaptureDeviceInput?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let audioDataOutput = AVCaptureAudioDataOutput()

    // Metal Compute & Render Pipeline
    let metalDevice: MTLDevice?
    private(set) var metalLibrary: MTLLibrary?
    private var commandQueue: MTLCommandQueue?
    private var computePipelineState: MTLComputePipelineState?
    private(set) var postGradePipelineState: MTLComputePipelineState?
    private var textureCache: CVMetalTextureCache?
    private var dummy3DLUTTexture: MTLTexture?
    private var custom3DLUTTexture: MTLTexture?

    private var outputBufferPool: CVPixelBufferPool?
    private var previewBufferPool: CVPixelBufferPool?
    private var currentPoolDimensions = CMVideoDimensions(width: 0, height: 0)

    private let previewLock = NSLock()
    private var _latestPreviewTexture: MTLTexture?
    var latestPreviewTexture: MTLTexture? {
        previewLock.lock()
        defer { previewLock.unlock() }
        return _latestPreviewTexture
    }

    // AVAssetWriter Recording Stack
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
        setupOrientationObserver()
    }

    // MARK: - Metal Initialization

    private func setupMetalPipeline() {
        guard let device = metalDevice else {
            statusMessage = "Metal GPU Unavailable"
            return
        }
        self.commandQueue = device.makeCommandQueue()
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

    private func setupOrientationObserver() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDeviceOrientationChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
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
            statusMessage = "录制中锁定画面方向"
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
            let (tex, name) = try CubeLUTManager.loadCubeLUT(from: url, device: device)
            self.custom3DLUTTexture = tex
            self.customLUTName = name
            self.grade.lookPreset = .customCube
            self.statusMessage = "已加载 3D LUT: \(name)"
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

            videoDataOutput.alwaysDiscardsLateVideoFrames = false
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

        } catch {
            DispatchQueue.main.async {
                self.statusMessage = "配置采集管线失败: \(error.localizedDescription)"
            }
        }
    }

    private func configureVideoOutputConnection() {
        guard let connection = videoDataOutput.connection(with: .video) else { return }
        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .landscapeRight
        }
        if connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = (selectedSensorPreset == .openGateMax) ? .off : .standard
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
            }
            try? self.selectBestSensorFormat(for: newCamera, preset: self.selectedSensorPreset, targetFPS: self.selectedFPS.rawValue)
            self.configureVideoOutputConnection()
            self.captureSession.commitConfiguration()
        }
    }

    private func reconfigureFormatAndSession() {
        guard !isRecording else { return }
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            self.captureSession.beginConfiguration()
            try? self.selectBestSensorFormat(for: device, preset: self.selectedSensorPreset, targetFPS: self.selectedFPS.rawValue)
            self.configureVideoOutputConnection()
            self.captureSession.commitConfiguration()
        }
    }

    private func selectBestSensorFormat(for device: AVCaptureDevice, preset: SensorFormatPreset, targetFPS: Double) throws {
        let tenBitSubtypes: [OSType] = [
            kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange, // 'x422'
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange  // 'x420'
        ]
        let eightBitSubtypes: [OSType] = [
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,   // '420f'
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange   // '420v'
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
            let dimA = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let dimB = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            let areaA = Int64(dimA.width) * Int64(dimA.height)
            let areaB = Int64(dimB.width) * Int64(dimB.height)
            if areaA != areaB {
                return areaA > areaB
            }
            let subA = CMFormatDescriptionGetMediaSubType(a.formatDescription)
            let subB = CMFormatDescriptionGetMediaSubType(b.formatDescription)
            let is10A = tenBitSubtypes.contains(subA) ? 1 : 0
            let is10B = tenBitSubtypes.contains(subB) ? 1 : 0
            if is10A != is10B {
                return is10A > is10B
            }
            let hlgA = a.supportedColorSpaces.contains(.HLG_BT2020) ? 1 : 0
            let hlgB = b.supportedColorSpaces.contains(.HLG_BT2020) ? 1 : 0
            return hlgA > hlgB
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
        let ratioStr = abs(Double(dims.width) / Double(max(1, dims.height)) - (4.0 / 3.0)) < 0.05 ? "Open Gate 4:3" : "16:9"
        let bitStr = isTenBit ? "10-Bit BT.2020" : "Wide P3 → 16-Bit Float"

        DispatchQueue.main.async {
            self.minISO = minISOVal
            self.maxISO = maxISOVal
            self.currentISO = currentISOVal
            self.activeResolutionBadge = "\(dims.width)×\(dims.height) (\(ratioStr)) @ \(Int(effectiveFPS))fps"
            self.activePipelineBadge = "\(self.selectedLens.focalLabel) • \(bitStr) → Apple Log"
            self.statusMessage = "感光元件满血输出已就绪"
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

    private func applyWhiteBalanceKelvin(_ kelvin: Float) {
        guard isWhiteBalanceLocked else { return }
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            let tempAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: kelvin, tint: 0.0)
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
                    let tempAndTint = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: self.whiteBalanceKelvin, tint: 0.0)
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

    // MARK: - 10-Bit ProRes / HEVC Recording Control

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        videoProcessingQueue.async { [weak self] in
            guard let self = self, !self.isRecording else { return }
            let width = Int(self.currentPoolDimensions.width > 0 ? self.currentPoolDimensions.width : 3840)
            let height = Int(self.currentPoolDimensions.height > 0 ? self.currentPoolDimensions.height : 2160)

            let fileName = "CineLog13Pro_\(width)x\(height)_\(Int(Date().timeIntervalSince1970)).mov"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try? FileManager.default.removeItem(at: outputURL)

            do {
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

                let colorProperties: [String: Any]
                if self.grade.bakeGradeIntoRecording && self.grade.lookPreset != .rawLog {
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

                var videoSettings: [String: Any] = [
                    AVVideoCodecKey: self.selectedCodec.avCodecType.rawValue,
                    AVVideoWidthKey: width,
                    AVVideoHeightKey: height,
                    AVVideoColorPropertiesKey: colorProperties
                ]

                if self.selectedCodec == .hevc10Bit {
                    videoSettings[AVVideoCompressionPropertiesKey] = [
                        AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                        AVVideoAverageBitRateKey: 140_000_000,
                        AVVideoExpectedSourceFrameRateKey: self.selectedFPS.rawValue
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

                if writer.canAdd(vInput) {
                    writer.add(vInput)
                }

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
                    throw writer.error ?? NSError(domain: "CineLog13Pro", code: -20, userInfo: [NSLocalizedDescriptionKey: "AVAssetWriter 启动失败"])
                }

                self.assetWriter = writer
                self.videoWriterInput = vInput
                self.audioWriterInput = aInput
                self.pixelBufferAdaptor = adaptor
                self.recordingStartTime = nil
                self.currentOutputURL = outputURL

                let modeDesc = self.grade.bakeGradeIntoRecording ? "直出调色 (\(self.grade.lookPreset.title))" : "纯净 Apple Log"
                DispatchQueue.main.async {
                    self.isRecording = true
                    self.recordingDuration = 0
                    self.statusMessage = "REC \(width)×\(height) • \(modeDesc)"
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusMessage = "录制初始化失败: \(error.localizedDescription)"
                }
            }
        }
    }

    private func stopRecording() {
        videoProcessingQueue.async { [weak self] in
            guard let self = self, self.isRecording, let writer = self.assetWriter else { return }
            self.isRecording = false
            self.videoWriterInput?.markAsFinished()
            self.audioWriterInput?.markAsFinished()

            let outputURL = self.currentOutputURL
            writer.finishWriting {
                DispatchQueue.main.async {
                    self.lastRecordedVideoURL = outputURL
                    self.statusMessage = "正在保存视频至系统相册..."
                }
                if let url = outputURL {
                    self.saveVideoToPhotoLibrary(url: url)
                }
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
        if currentPoolDimensions.width == width && currentPoolDimensions.height == height &&
           outputBufferPool != nil && previewBufferPool != nil {
            return
        }
        currentPoolDimensions = CMVideoDimensions(width: width, height: height)

        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 6]
        let bufferAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &outputBufferPool)
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &previewBufferPool)
    }
}

// MARK: - 5. AVCaptureVideoDataOutput & Audio Sample Buffer Delegate

extension AppleLogCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

    struct ColorGradeUniforms {
        var isNativeAppleLog: Int32
        var isFullRangeYCbCr: Int32
        var isSDRSensorInput: Int32
        var rotationMode: Int32
        var mirrorHorizontal: Int32

        var lookPreset: Int32
        var bakeGradeIntoRecording: Int32
        var monitorDiagnosticMode: Int32
        var hasCustom3DLUT: Int32

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
        var filmGrain: Float
        var frameTime: Float

        var liftRGB: SIMD4<Float>
        var gammaRGB: SIMD4<Float>
        var gainRGB: SIMD4<Float>
    }

    func makeShaderUniforms() -> ColorGradeUniforms {
        let g = grade
        return ColorGradeUniforms(
            isNativeAppleLog: hardwareSupportsNativeAppleLog ? 1 : 0,
            isFullRangeYCbCr: activeFormatIsFullRange ? 1 : 0,
            isSDRSensorInput: activeFormatIsSDRFallback ? 1 : 0,
            rotationMode: Int32(orientationMode.rawValue),
            mirrorHorizontal: isMirroredHorizontal ? 1 : 0,
            lookPreset: Int32(g.lookPreset.rawValue),
            bakeGradeIntoRecording: g.bakeGradeIntoRecording ? 1 : 0,
            monitorDiagnosticMode: Int32(monitorDiagnostic.rawValue),
            hasCustom3DLUT: custom3DLUTTexture != nil ? 1 : 0,
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
            filmGrain: g.filmGrain,
            frameTime: Float(CACurrentMediaTime() - startTimestamp),
            liftRGB: SIMD4<Float>(g.liftR, g.liftG, g.liftB, 0.0),
            gammaRGB: SIMD4<Float>(g.gammaR, g.gammaG, g.gammaB, 0.0),
            gainRGB: SIMD4<Float>(g.gainR, g.gainG, g.gainB, 1.0)
        )
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if output == audioDataOutput {
            if isRecording,
               let writer = assetWriter,
               writer.status == .writing,
               recordingStartTime != nil,
               let aInput = audioWriterInput,
               aInput.isReadyForMoreMediaData {
                aInput.append(sampleBuffer)
            }
            return
        }

        guard let inputPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let cache = textureCache,
              let commandQueue = commandQueue,
              let pipelineState = computePipelineState,
              let fallbackLUT = dummy3DLUTTexture else {
            return
        }

        let sensorW = CVPixelBufferGetWidth(inputPixelBuffer)
        let sensorH = CVPixelBufferGetHeight(inputPixelBuffer)

        let outW = orientationMode.isPortraitSwapped ? sensorH : sensorW
        let outH = orientationMode.isPortraitSwapped ? sensorW : sensorH
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
            return
        }

        var logOutPixelBuffer: CVPixelBuffer?
        var previewPixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outPool, &logOutPixelBuffer)
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, prevPool, &previewPixelBuffer)

        guard let logOutPB = logOutPixelBuffer, let prevOutPB = previewPixelBuffer else { return }

        var cvLogOutTex: CVMetalTexture?
        var cvPrevOutTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, logOutPB, nil, .rgba16Float, outW, outH, 0, &cvLogOutTex
        )
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, prevOutPB, nil, .rgba16Float, outW, outH, 0, &cvPrevOutTex
        )

        guard let logOutRef = cvLogOutTex, let prevOutRef = cvPrevOutTex,
              let logOutTex = CVMetalTextureGetTexture(logOutRef),
              let prevOutTex = CVMetalTextureGetTexture(prevOutRef),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        var uniforms = makeShaderUniforms()

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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        previewLock.lock()
        _latestPreviewTexture = prevOutTex
        previewLock.unlock()

        if isRecording,
           let writer = assetWriter,
           writer.status == .writing,
           let vInput = videoWriterInput,
           let adaptor = pixelBufferAdaptor {
            if recordingStartTime == nil {
                recordingStartTime = timestamp
                writer.startSession(atSourceTime: timestamp)
            }
            if vInput.isReadyForMoreMediaData {
                adaptor.append(logOutPB, withPresentationTime: timestamp)
            }
            if let start = recordingStartTime {
                let elapsed = CMTimeGetSeconds(CMTimeSubtract(timestamp, start))
                DispatchQueue.main.async {
                    self.recordingDuration = max(0, elapsed)
                }
            }
        }

        frameCounter += 1
        if frameCounter % 8 == 0 {
            computeFastLumaHistogram(from: inputPixelBuffer, isTenBit: activeFormatIsTenBit)
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
            for y in stride(from: 0, to: height, by: 20) {
                let rowOffset = y * strideU16
                for x in stride(from: 0, to: width, by: 20) {
                    let normalized = Float(uint16Ptr[rowOffset + x]) / 65535.0
                    let binIdx = min(63, max(0, Int(normalized * 63.0)))
                    bins[binIdx] += 1
                    totalSamples += 1
                }
            }
        } else {
            let uint8Ptr = baseAddr.assumingMemoryBound(to: UInt8.self)
            for y in stride(from: 0, to: height, by: 20) {
                let rowOffset = y * bytesPerRow
                for x in stride(from: 0, to: width, by: 20) {
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
                let width = Int(abs(naturalSize.width))
                let height = Int(abs(naturalSize.height))

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
                        var uniforms = self.makeShaderUniforms()
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

// MARK: - 6. EDR 10-Bit Metal Viewfinder (Aspect-Fit Scaling for Both Portrait & Landscape)

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

// MARK: - 7. Post-Production Color Grading Studio Sheet (后期调色室)

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
    @State private var exportStatus: String = "可从相册选择已拍摄的 Log 视频或直接调整上一条录制素材"
    @State private var showCubeShareSheet: Bool = false
    @State private var exportedCubeURL: URL?

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black)
                        .frame(height: 250)

                    if let player = player {
                        VideoPlayer(player: player)
                            .frame(height: 250)
                            .cornerRadius(12)
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "film.stack")
                                .font(.system(size: 38))
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
                        Label("导出 .cube LUT", systemImage: "cube.transparent")
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
                                exportStatus = "渲染完成！10-Bit 调色成片已存入系统相册"
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

                Text(exportStatus)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.yellow)

                ScrollView {
                    ColorGradingControlPanel(engine: engine, showCubeImporter: .constant(false))
                        .padding(.horizontal, 16)
                }
            }
            .background(Color(white: 0.08).ignoresSafeArea())
            .navigationTitle("后期调色室 (Post Grading & LUT)")
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
                    exportStatus = "已自动载入最新录制的 Apple Log 素材"
                }
            }
            .onChange(of: selectedItem) { newItem in
                guard let newItem = newItem else { return }
                Task {
                    if let movie = try? await newItem.loadTransferable(type: VideoMovieTransferable.self) {
                        await MainActor.run {
                            activeVideoURL = movie.url
                            player = AVPlayer(url: movie.url)
                            exportStatus = "已载入视频，调整下方参数后点击「渲染导出 10-Bit」即可生成调色成片"
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

// MARK: - 8. Reusable Live & Post Color Grading Control Panel

struct ColorGradingControlPanel: View {
    @ObservedObject var engine: AppleLogCameraEngine
    @Binding var showCubeImporter: Bool
    @State private var gradingSubTab: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(LookPresetOption.allCases) { preset in
                        Button {
                            engine.grade.lookPreset = preset
                        } label: {
                            Text(preset.title)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(engine.grade.lookPreset == preset ? Color.yellow : Color.white.opacity(0.12))
                                .foregroundColor(engine.grade.lookPreset == preset ? .black : .white)
                                .cornerRadius(6)
                        }
                    }

                    Button {
                        showCubeImporter = true
                    } label: {
                        Label("导入 .cube", systemImage: "folder.badge.plus")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Color.cyan.opacity(0.25))
                            .foregroundColor(.cyan)
                            .cornerRadius(6)
                    }
                }
            }

            HStack {
                Toggle(isOn: $engine.grade.bakeGradeIntoRecording) {
                    Text(engine.grade.bakeGradeIntoRecording ? "录制模式：将实时调色烧录直出" : "录制模式：录制纯净 Apple Log（调色仅监看）")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(engine.grade.bakeGradeIntoRecording ? .orange : .green)
                }
                .tint(.orange)

                Button("重置") {
                    engine.grade.reset()
                }
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)
            }

            Picker("调色分组", selection: $gradingSubTab) {
                Text("一级调色 (Primary)").tag(0)
                Text("三路色轮 (Lift/Gamma/Gain)").tag(1)
                Text("高光滚降 & 胶片颗粒").tag(2)
            }
            .pickerStyle(.segmented)

            if gradingSubTab == 0 {
                VStack(spacing: 6) {
                    GradeSliderRow(label: "曝光 EV", value: $engine.grade.exposureEV, range: -2.5...2.5, format: "%+.1f EV", tint: .yellow)
                    GradeSliderRow(label: "对比度", value: $engine.grade.contrast, range: 0.5...1.6, format: "%.2f", tint: .white)
                    GradeSliderRow(label: "饱和度", value: $engine.grade.saturation, range: 0.0...2.0, format: "%.2f", tint: .pink)
                    GradeSliderRow(label: "自然饱和", value: $engine.grade.vibrance, range: -1.0...1.0, format: "%+.2f", tint: .pink)
                    GradeSliderRow(label: "色温冷暖", value: $engine.grade.temperatureOffset, range: -1.0...1.0, format: "%+.2f", tint: .orange)
                    GradeSliderRow(label: "色调绿品", value: $engine.grade.tintOffset, range: -1.0...1.0, format: "%+.2f", tint: .green)
                }
            } else if gradingSubTab == 1 {
                VStack(spacing: 6) {
                    GradeSliderRow(label: "暗部 Lift R", value: $engine.grade.liftR, range: -0.2...0.2, format: "%+.2f", tint: .red)
                    GradeSliderRow(label: "暗部 Lift G", value: $engine.grade.liftG, range: -0.2...0.2, format: "%+.2f", tint: .green)
                    GradeSliderRow(label: "暗部 Lift B", value: $engine.grade.liftB, range: -0.2...0.2, format: "%+.2f", tint: .blue)
                    GradeSliderRow(label: "中灰 Gamma B", value: $engine.grade.gammaB, range: -0.3...0.3, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "高光 Gain R", value: $engine.grade.gainR, range: 0.7...1.3, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "高光 Gain B", value: $engine.grade.gainB, range: 0.7...1.3, format: "%.2f", tint: .blue)
                }
            } else {
                VStack(spacing: 6) {
                    GradeSliderRow(label: "高光滚降", value: $engine.grade.highlightRollOff, range: 0.0...1.0, format: "%.2f", tint: .orange)
                    GradeSliderRow(label: "暗部提亮", value: $engine.grade.shadowLift, range: -0.25...0.25, format: "%+.2f", tint: .cyan)
                    GradeSliderRow(label: "反差中心", value: $engine.grade.pivot, range: 0.30...0.65, format: "%.3f", tint: .white)
                    GradeSliderRow(label: "LUT 浓度", value: $engine.grade.lutIntensity, range: 0.0...1.0, format: "%.2f", tint: .yellow)
                    GradeSliderRow(label: "胶片颗粒", value: $engine.grade.filmGrain, range: 0.0...1.0, format: "%.2f", tint: .gray)
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
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
                .frame(width: 82, alignment: .leading)
            Slider(value: $value, in: range)
                .tint(tint)
            Text(String(format: format, value))
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(tint)
                .frame(width: 58, alignment: .trailing)
        }
    }
}

// MARK: - 9. Ergonomic Cinema Camera HUD & Collapsible Control Drawer

enum ActiveControlTab: Int, CaseIterable, Identifiable {
    case none     = 0
    case format   = 1
    case exposure = 2
    case grading  = 3
    case monitor  = 4

    var id: Int { rawValue }
}

struct ViewfinderGuidesOverlay: View {
    let showGrid: Bool
    let aspectGuide: AspectGuideOption

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            ZStack {
                if showGrid {
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
                    .stroke(Color.white.opacity(0.22), lineWidth: 0.8)
                }

                if aspectGuide == .scope239 {
                    let targetH = min(h, w / 2.39)
                    Rectangle()
                        .stroke(Color.yellow.opacity(0.65), lineWidth: 1.2)
                        .frame(width: w * 0.94, height: targetH * 0.94)
                } else if aspectGuide == .cinema185 {
                    let targetH = min(h, w / 1.85)
                    Rectangle()
                        .stroke(Color.yellow.opacity(0.65), lineWidth: 1.2)
                        .frame(width: w * 0.94, height: targetH * 0.94)
                } else if aspectGuide == .vertical916 {
                    let targetW = min(w, h * (9.0 / 16.0))
                    Rectangle()
                        .stroke(Color.cyan.opacity(0.65), lineWidth: 1.2)
                        .frame(width: targetW * 0.92, height: h * 0.92)
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
                    .frame(width: 1.8, height: max(2, CGFloat(bins[index]) * 32))
            }
        }
        .padding(5)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.white.opacity(0.2), lineWidth: 0.8)
        )
    }
}

struct CineLogCameraView: View {
    @StateObject private var engine = AppleLogCameraEngine()
    @State private var activeTab: ActiveControlTab = .none
    @State private var showPostGradingStudio: Bool = false
    @State private var showCubeImporter: Bool = false
    @State private var showCubeShareSheet: Bool = false
    @State private var exportedCubeURL: URL?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            MetalLogViewfinder(engine: engine)
                .ignoresSafeArea()
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        activeTab = .none
                    }
                }

            ViewfinderGuidesOverlay(showGrid: engine.showGridOverlay, aspectGuide: engine.aspectGuide)
                .ignoresSafeArea()

            VStack(spacing: 8) {
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(engine.selectedSensorPreset.shortTitle)
                                .font(.system(size: 10, weight: .black, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.yellow)
                                .foregroundColor(.black)
                                .cornerRadius(4)

                            Text(engine.activeResolutionBadge)
                                .font(.system(size: 11, weight: .bold, design: .monospaced))
                                .foregroundColor(.white)
                        }
                        Text("\(engine.activePipelineBadge) • \(engine.statusMessage)")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundColor(.white.opacity(0.8))
                            .lineLimit(1)
                    }

                    Spacer()

                    Button {
                        engine.rotateOrientation90Degrees()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: engine.orientationMode.isPortraitSwapped ? "rectangle.portrait" : "rectangle")
                            Text(engine.orientationMode.shortLabel)
                        }
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(.yellow)
                        .cornerRadius(6)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.yellow.opacity(0.5), lineWidth: 1))
                    }

                    Button {
                        showPostGradingStudio = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "slider.horizontal.below.rectangle")
                            Text("后期调色")
                        }
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(Color.cyan.opacity(0.25))
                        .foregroundColor(.cyan)
                        .cornerRadius(6)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan.opacity(0.5), lineWidth: 1))
                    }

                    LumaHistogramOverlay(bins: engine.lumaHistogram)
                }
                .padding(.horizontal, 12)
                .padding(.top, 6)

                Spacer()

                HStack(spacing: 12) {
                    ForEach(CameraLensOption.allCases) { lens in
                        Button {
                            engine.selectedLens = lens
                        } label: {
                            Text(lens.rawValue)
                                .font(.system(size: 12, weight: .bold, design: .monospaced))
                                .frame(width: 40, height: 40)
                                .background(engine.selectedLens == lens ? Color.yellow : Color.black.opacity(0.65))
                                .foregroundColor(engine.selectedLens == lens ? .black : .white)
                                .clipShape(Circle())
                                .overlay(Circle().stroke(Color.white.opacity(0.25), lineWidth: 1))
                        }
                    }
                }
                .padding(.bottom, 2)

                if activeTab != .none {
                    VStack(spacing: 10) {
                        if activeTab == .format {
                            FormatAndOpenGateDrawer(engine: engine)
                        } else if activeTab == .exposure {
                            ExposureAndWBDrawer(engine: engine)
                        } else if activeTab == .grading {
                            ColorGradingControlPanel(engine: engine, showCubeImporter: $showCubeImporter)
                        } else if activeTab == .monitor {
                            MonitorToolsDrawer(
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
                    .background(Color.black.opacity(0.84))
                    .cornerRadius(14)
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.16), lineWidth: 1))
                    .padding(.horizontal, 12)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        DrawerTabButton(
                            title: "画幅/OpenGate",
                            icon: "aspectratio",
                            isActive: activeTab == .format
                        ) { toggleTab(.format) }

                        DrawerTabButton(
                            title: "曝光/快门",
                            icon: "camera.aperture",
                            isActive: activeTab == .exposure
                        ) { toggleTab(.exposure) }

                        DrawerTabButton(
                            title: "实时调色/LUT",
                            icon: "paintpalette",
                            isActive: activeTab == .grading
                        ) { toggleTab(.grading) }

                        DrawerTabButton(
                            title: "假色/斑马纹",
                            icon: "waveform.path.ecg",
                            isActive: activeTab == .monitor
                        ) { toggleTab(.monitor) }
                    }
                    .padding(.horizontal, 12)

                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(formatTimecode(engine.recordingDuration, fps: engine.selectedFPS.rawValue))
                                .font(.system(size: 16, weight: .bold, design: .monospaced))
                                .foregroundColor(engine.isRecording ? .red : .white)
                            Text("\(engine.selectedCodec.rawValue) • \(engine.grade.lookPreset.title)")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundColor(.yellow)
                        }
                        .frame(width: 145, alignment: .leading)

                        Spacer()

                        Button(action: { engine.toggleRecording() }) {
                            ZStack {
                                Circle()
                                    .stroke(Color.white, lineWidth: 3)
                                    .frame(width: 68, height: 68)
                                RoundedRectangle(cornerRadius: engine.isRecording ? 8 : 30)
                                    .fill(Color.red)
                                    .frame(
                                        width: engine.isRecording ? 28 : 54,
                                        height: engine.isRecording ? 28 : 54
                                    )
                                    .animation(.easeInOut(duration: 0.2), value: engine.isRecording)
                            }
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 2) {
                            Text(engine.grade.bakeGradeIntoRecording ? "REC: 调色直出" : "REC: 纯净 APPLE LOG")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(engine.grade.bakeGradeIntoRecording ? .orange : .green)
                            Text("方向: \(engine.orientationMode.shortLabel)")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundColor(.gray)
                        }
                        .frame(width: 145, alignment: .trailing)
                    }
                    .padding(.horizontal, 18)
                    .padding(.bottom, 10)
                }
                .padding(.top, 8)
                .background(Color.black.opacity(0.78))
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
        let totalSeconds = Int(seconds)
        let mins = (totalSeconds / 60) % 60
        let secs = totalSeconds % 60
        let frames = Int((seconds - Double(totalSeconds)) * fps)
        return String(format: "%02d:%02d:%02d", mins, secs, max(0, frames))
    }
}

// MARK: - 10. Sub-Drawer Views (Format/OpenGate, Exposure/WB, Monitor Tools)

struct DrawerTabButton: View {
    let title: String
    let icon: String
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                Text(title)
            }
            .font(.system(size: 11, weight: .bold, design: .monospaced))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(isActive ? Color.yellow : Color.white.opacity(0.12))
            .foregroundColor(isActive ? .black : .white)
            .cornerRadius(8)
        }
    }
}

struct FormatAndOpenGateDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("传感器画幅与最大分辨率 (Open Gate / 4K)")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)

            ForEach(SensorFormatPreset.allCases) { preset in
                Button {
                    engine.selectedSensorPreset = preset
                } label: {
                    HStack {
                        Image(systemName: engine.selectedSensorPreset == preset ? "checkmark.circle.fill" : "circle")
                        Text(preset.rawValue)
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(engine.selectedSensorPreset == preset ? Color.yellow.opacity(0.22) : Color.white.opacity(0.08))
                    .foregroundColor(engine.selectedSensorPreset == preset ? .yellow : .white)
                    .cornerRadius(8)
                }
            }

            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("编码格式 (Codec)")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(.gray)
                    Picker("Codec", selection: $engine.selectedCodec) {
                        ForEach(RecordingCodecOption.allCases) { codec in
                            Text(codec.rawValue).tag(codec)
                        }
                    }
                    .pickerStyle(.segmented)
                }
            }

            HStack(spacing: 10) {
                Text("帧率:")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                Picker("FPS", selection: $engine.selectedFPS) {
                    ForEach(FrameRateOption.allCases) { fps in
                        Text(fps.label).tag(fps)
                    }
                }
                .pickerStyle(.segmented)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("拍摄画面方向:")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                    Spacer()
                    Toggle("陀螺仪自动旋转", isOn: $engine.autoFollowDeviceOrientation)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(.cyan)
                        .tint(.cyan)
                }
                Picker("Orientation", selection: $engine.orientationMode) {
                    ForEach(CaptureOrientationMode.allCases) { mode in
                        Text(mode.shortLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
    }
}

struct ExposureAndWBDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Toggle("手动电影曝光 (ISO + 快门开角)", isOn: $engine.isManualExposure)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.yellow)
                    .tint(.yellow)
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
                Toggle("锁定白平衡色温 (Kelvin)", isOn: $engine.isWhiteBalanceLocked)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.cyan)
                    .tint(.cyan)
            }

            if engine.isWhiteBalanceLocked {
                GradeSliderRow(
                    label: "色温 (K)",
                    value: $engine.whiteBalanceKelvin,
                    range: 2800...8500,
                    format: "%.0fK",
                    tint: .cyan
                )
            }
        }
    }
}

struct MonitorToolsDrawer: View {
    @ObservedObject var engine: AppleLogCameraEngine
    let onExportCube: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("专业曝光诊断与构图辅助")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.yellow)

            Picker("Diagnostic", selection: $engine.monitorDiagnostic) {
                ForEach(MonitorDiagnosticMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Text("电影遮幅框:")
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

                Toggle("九宫格", isOn: $engine.showGridOverlay)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .tint(.yellow)
            }

            HStack {
                Toggle("水平镜像画面", isOn: $engine.isMirroredHorizontal)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .tint(.cyan)

                Spacer()

                Button(action: onExportCube) {
                    Label("导出当前调色为 .cube LUT", systemImage: "square.and.arrow.up")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.yellow)
                        .foregroundColor(.black)
                        .cornerRadius(6)
                }
            }
        }
    }
}

// MARK: - 11. App Entry Point

@main
struct CineLog13ProApp: App {
    var body: some Scene {
        WindowGroup {
            CineLogCameraView()
                .preferredColorScheme(.dark)
        }
    }
}