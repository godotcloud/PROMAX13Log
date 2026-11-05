//
//  iPhone13ProMaxLogCameraApp.swift
//  CineLog13Pro - 10-Bit Apple Log Camera for iPhone 13 Pro Max
//
//  Target: iOS 16.0+ (Optimized for iPhone 13 Pro Max A15 Bionic & compatible with iPhone 15 Pro+)
//  Frameworks: UIKit, SwiftUI, AVFoundation, Metal, MetalKit, CoreVideo, CoreMedia, VideoToolbox, Photos, QuartzCore
//  GitHub Actions Compatible: Single-file drop-in source with embedded runtime-compiled Metal compute + render shaders.
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
import QuartzCore
import Combine

// MARK: - 1. Embedded Metal Compute & Render Shaders (Compiled at Runtime)

enum MetalShaderLibrarySource {
    /// Complete Metal Shading Language (MSL) source compiled at runtime via device.makeLibrary(source:options:)
    /// Includes:
    /// 1) `processAppleLogFrame` Compute Kernel: 10-bit BiPlanar YCbCr ('x422'/'x420') -> BT.2020 HLG Inverse OETF -> Linear -> Apple Log OETF
    /// 2) `vertexViewfinder` & `fragmentViewfinder` Render Shaders: Aspect-fit scaling from 4K rgba16Float to iPhone 13 Pro Max bgr10a2Unorm EDR display
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct LogPipelineUniforms {
        int isNativeAppleLog;     // 1 if hardware outputs native Apple Log (iPhone 15 Pro+), 0 if HLG_BT2020 (iPhone 13 Pro Max)
        int previewLUTMode;       // 0 = Raw Apple Log, 1 = Apple Log -> Rec.709 Preview LUT, 2 = False Color IRE Map
        int showZebraOverlay;     // 1 = Show 94% IRE highlight clipping zebras
        float exposureBiasEV;     // Exposure bias in linear light (-2.5 to +2.5 EV)
        float highlightRollOff;   // Highlight shoulder compression factor (0.0 to 1.0)
        float frameTime;          // Timestamp for zebra stripe animation
    };

    // --- ITU-R BT.2020 Video-Range 10-bit YCbCr ('x420' / 'x422') to Normalized R'G'B' ---
    inline float3 ycbcr10VideoRangeToBT2020RGB(float yRaw, float2 cbcrRaw) {
        // CVMetalTextureCache maps 10-bit bi-planar video range into [0.0, 1.0] across the 16-bit word
        float y10  = yRaw * 1023.0;
        float2 c10 = cbcrRaw * 1023.0;

        float y  = clamp((y10 - 64.0) / (940.0 - 64.0), 0.0, 1.0);
        float cb = (c10.x - 512.0) / (960.0 - 64.0);
        float cr = (c10.y - 512.0) / (960.0 - 64.0);

        // ITU-R BT.2020 Non-Constant Luminance (NCL) matrix
        float r = y + 1.4746 * cr;
        float g = y - 0.16455312684366 * cb - 0.57135312684366 * cr;
        float b = y + 1.8814 * cb;

        return clamp(float3(r, g, b), 0.0, 1.0);
    }

    // --- ITU-R BT.2100 HLG Inverse OETF (Non-linear HLG E' -> Scene-Linear Light E) ---
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
        // Scale HLG scene-referred diffuse white to Apple Log 18% gray anchor (R = 0.18 -> P = 0.488)
        float gain = exp2(evBias) * 2.4;
        sceneLinear *= gain;

        // Optional filmic highlight roll-off to smooth A15 Bionic LTM shoulder
        if (rollOff > 0.001) {
            float3 shoulder = sceneLinear / (1.0 + sceneLinear * (0.25 * rollOff));
            sceneLinear = mix(sceneLinear, shoulder, rollOff);
        }
        return sceneLinear;
    }

    // --- Official Apple Log Profile White Paper Encoding Function (Linear Scene Reflection R -> Apple Log P) ---
    // Constants from Apple Log Profile White Paper (September 2023):
    //   R_0 = -0.05641088, R_t = 0.01, c = 47.28711236,
    //   beta = 0.00964052, gamma = 0.08550479, delta = 0.69336945
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

    // --- Official Apple Log Profile White Paper Decoding Function (Apple Log P -> Linear Scene Reflection R) ---
    inline float appleLogToLinearChannel(float P) {
        const float R_0   = -0.05641088;
        const float R_t   =  0.01;
        const float c     =  47.28711236;
        const float beta  =  0.00964052;
        const float gamma =  0.08550479;
        const float delta =  0.69336945;
        const float P_t   =  0.20856507; // c * (R_t - R_0)^2

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

    // --- Apple Log (BT.2020) -> Rec.709 Display-Referred 3D/Analytical LUT ---
    inline float3 appleLogToRec709Preview(float3 appleLogRGB) {
        float3 lin2020 = appleLogToLinear(appleLogRGB);

        // ITU-R BT.2020 Linear to ITU-R BT.709 Linear 3x3 Matrix
        float3 lin709;
        lin709.r =  1.6604910 * lin2020.r - 0.5876411 * lin2020.g - 0.0728499 * lin2020.b;
        lin709.g = -0.1245505 * lin2020.r + 1.1328999 * lin2020.g - 0.0083494 * lin2020.b;
        lin709.b = -0.0181508 * lin2020.r - 0.1005789 * lin2020.g + 1.1187297 * lin2020.b;
        lin709 = max(lin709, 0.0);

        // ACES-inspired filmic S-curve tone-map for natural Rec.709 monitoring
        float3 x = lin709 * 1.35;
        float3 mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        mapped = clamp(mapped, 0.0, 1.0);

        return pow(mapped, float3(1.0 / 2.2));
    }

    // --- False Color IRE Exposure Diagnostic Map (Calibrated to Apple Log 18% Gray = 0.488 IRE) ---
    inline float3 falseColorMap(float lumaIRE) {
        if (lumaIRE < 0.05) return float3(0.55, 0.0, 0.85);       // Purple: Crushed Shadows (<5%)
        else if (lumaIRE < 0.18) return float3(0.1, 0.35, 0.95);  // Blue: Deep Shadows
        else if (lumaIRE < 0.44) return float3(0.2, 0.75, 0.85);  // Cyan: Lower Midtones
        else if (lumaIRE < 0.53) return float3(0.15, 0.85, 0.25); // Green: 18% Middle Gray Zone (0.488 in Apple Log)
        else if (lumaIRE < 0.62) return float3(0.95, 0.55, 0.65); // Pink: +1 EV Skin Tone Zone
        else if (lumaIRE < 0.82) return float3(0.8, 0.8, 0.8);    // Neutral Gray: Upper Midtones / Highlights
        else if (lumaIRE < 0.94) return float3(0.98, 0.85, 0.1);  // Yellow: Near Clipping Warning
        else return float3(0.98, 0.1, 0.1);                       // Red: Clipped Highlight (>94%)
    }

    // --- Compute Kernel: Dual Output (1. Recorded Apple Log Buffer, 2. Viewfinder Preview Texture) ---
    kernel void processAppleLogFrame(
        texture2d<float, access::read>  lumaPlane     [[texture(0)]],
        texture2d<float, access::read>  chromaPlane   [[texture(1)]],
        texture2d<float, access::write> logOutputTex  [[texture(2)]],
        texture2d<float, access::write> previewOutTex [[texture(3)]],
        constant LogPipelineUniforms&   uniforms      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= logOutputTex.get_width() || gid.y >= logOutputTex.get_height()) {
            return;
        }

        uint2 chromaCoord = uint2(
            gid.x * chromaPlane.get_width() / lumaPlane.get_width(),
            gid.y * chromaPlane.get_height() / lumaPlane.get_height()
        );

        float  ySample    = lumaPlane.read(gid).r;
        float2 cbcrSample = chromaPlane.read(chromaCoord).rg;

        float3 decodedRGB = ycbcr10VideoRangeToBT2020RGB(ySample, cbcrSample);

        // Step 1: Produce canonical Apple Log (BT.2020) signal
        float3 appleLogRGB;
        if (uniforms.isNativeAppleLog == 1) {
            appleLogRGB = decodedRGB;
        } else {
            float3 linearBT2020 = hlgToLinearBT2020(decodedRGB, uniforms.exposureBiasEV, uniforms.highlightRollOff);
            appleLogRGB = linearToAppleLog(linearBT2020);
        }

        // Write pure Apple Log RGB into the recording buffer (16-bit float target)
        logOutputTex.write(float4(appleLogRGB, 1.0), gid);

        // Step 2: Generate Viewfinder Monitoring Signal (Raw Log / Rec.709 LUT / False Color)
        float lumaLog = dot(appleLogRGB, float3(0.2627, 0.6780, 0.0593));
        float3 monitorRGB = appleLogRGB;

        if (uniforms.previewLUTMode == 1) {
            monitorRGB = appleLogToRec709Preview(appleLogRGB);
        } else if (uniforms.previewLUTMode == 2) {
            monitorRGB = falseColorMap(lumaLog);
        }

        if (uniforms.showZebraOverlay == 1 && lumaLog > 0.94) {
            int stripe = int(float(gid.x + gid.y) * 0.25 + uniforms.frameTime * 15.0) % 2;
            if (stripe == 0) {
                monitorRGB = float3(1.0, 0.15, 0.15);
            }
        }

        previewOutTex.write(float4(monitorRGB, 1.0), gid);
    }

    // --- Render Pipeline Shaders for MTKView (Scales 4K Texture to Any Screen Resolution & PixelFormat) ---
    struct VertexOut {
        float4 position [[position]];
        float2 texCoord;
    };

    struct ViewfinderScaleUniforms {
        float2 scale; // Aspect-fit XY scale factors
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

// MARK: - 2. Pipeline Configuration Models

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

enum PreviewLUTMode: Int, CaseIterable, Identifiable {
    case rawLog     = 0
    case rec709LUT  = 1
    case falseColor = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .rawLog:     return "Raw Log"
        case .rec709LUT:  return "Rec.709 LUT"
        case .falseColor: return "False Color"
        }
    }
}

enum FrameRateOption: Double, CaseIterable, Identifiable {
    case fps24 = 24.0
    case fps25 = 25.0
    case fps30 = 30.0
    case fps60 = 60.0

    var id: Double { rawValue }
    var label: String { "\(Int(rawValue)) fps" }
}

// MARK: - 3. Core 10-Bit Apple Log Capture & Metal Recording Engine

final class AppleLogCameraEngine: NSObject, ObservableObject {
    @Published var isSessionRunning: Bool = false
    @Published var isRecording: Bool = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var activePipelineBadge: String = "Initializing 10-Bit Pipeline..."
    @Published var statusMessage: String = "Ready"

    // User Controls
    @Published var selectedCodec: RecordingCodecOption = .proRes422
    @Published var selectedFPS: FrameRateOption = .fps24 {
        didSet { updateFrameRateAndShutter() }
    }
    @Published var previewMode: PreviewLUTMode = .rec709LUT
    @Published var isZebraEnabled: Bool = false
    @Published var exposureBiasEV: Float = 0.0
    @Published var highlightRollOff: Float = 0.35

    // Manual Cinema Exposure & White Balance Controls
    @Published var isManualExposure: Bool = false {
        didSet { updateFrameRateAndShutter() }
    }
    @Published var currentISO: Float = 100.0
    @Published var minISO: Float = 50.0
    @Published var maxISO: Float = 1600.0
    @Published var shutterAngleDegrees: Double = 180.0 {
        didSet { updateFrameRateAndShutter() }
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
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let audioDataOutput = AVCaptureAudioDataOutput()

    // Metal Compute & Render Pipeline
    let metalDevice: MTLDevice?
    private(set) var metalLibrary: MTLLibrary?
    private var commandQueue: MTLCommandQueue?
    private var computePipelineState: MTLComputePipelineState?
    private var textureCache: CVMetalTextureCache?
    private var outputBufferPool: CVPixelBufferPool?
    private var previewBufferPool: CVPixelBufferPool?
    private var currentPoolDimensions = CMVideoDimensions(width: 0, height: 0)

    // Thread-safe access to the latest rendered preview texture for MTKView
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

    // Pipeline flags
    private var hardwareSupportsNativeAppleLog: Bool = false
    private var frameCounter: UInt64 = 0
    private let startTimestamp: CFTimeInterval = CACurrentMediaTime()

    override init() {
        self.metalDevice = MTLCreateSystemDefaultDevice()
        super.init()
        setupMetalPipeline()
    }

    // MARK: - Metal Initialization

    private func setupMetalPipeline() {
        guard let device = metalDevice else {
            statusMessage = "Metal GPU Unavailable"
            return
        }
        self.commandQueue = device.makeCommandQueue()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)

        do {
            let library = try device.makeLibrary(source: MetalShaderLibrarySource.source, options: nil)
            self.metalLibrary = library
            guard let kernelFunc = library.makeFunction(name: "processAppleLogFrame") else {
                statusMessage = "Shader Kernel Missing"
                return
            }
            self.computePipelineState = try device.makeComputePipelineState(function: kernelFunc)
        } catch {
            statusMessage = "Shader Compile Error: \(error.localizedDescription)"
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

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            DispatchQueue.main.async { self.statusMessage = "未检测到后置广角主摄" }
            return
        }
        self.videoDevice = camera

        do {
            let videoInput = try AVCaptureDeviceInput(device: camera)
            if captureSession.canAddInput(videoInput) {
                captureSession.addInput(videoInput)
            }

            if includeAudio,
               let mic = AVCaptureDevice.default(for: .audio),
               let audioInput = try? AVCaptureDeviceInput(device: mic),
               captureSession.canAddInput(audioInput) {
                captureSession.addInput(audioInput)
            }

            try selectBest10BitLogFormat(for: camera, targetWidth: 3840, targetFPS: selectedFPS.rawValue)

            videoDataOutput.alwaysDiscardsLateVideoFrames = false
            let availablePixelFormats = videoDataOutput.availableVideoPixelFormatTypes
            if availablePixelFormats.contains(kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange) {
                videoDataOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
                ]
            } else if availablePixelFormats.contains(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange) {
                videoDataOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                ]
            }

            videoDataOutput.setSampleBufferDelegate(self, queue: videoProcessingQueue)
            if captureSession.canAddOutput(videoDataOutput) {
                captureSession.addOutput(videoDataOutput)
            }

            if let connection = videoDataOutput.connection(with: .video),
               connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .standard
            }

            audioDataOutput.setSampleBufferDelegate(self, queue: videoProcessingQueue)
            if captureSession.canAddOutput(audioDataOutput) {
                captureSession.addOutput(audioDataOutput)
            }

        } catch {
            DispatchQueue.main.async {
                self.statusMessage = "配置采集管线失败: \(error.localizedDescription)"
            }
        }
    }

    private func selectBest10BitLogFormat(for device: AVCaptureDevice, targetWidth: Int32, targetFPS: Double) throws {
        let tenBitSubtypes: [OSType] = [
            kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange, // 'x422'
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange  // 'x420'
        ]

        let candidates = device.formats.filter { format in
            let desc = format.formatDescription
            let dims = CMVideoFormatDescriptionGetDimensions(desc)
            let subType = CMFormatDescriptionGetMediaSubType(desc)
            let supportsFPS = format.videoSupportedFrameRateRanges.contains {
                $0.minFrameRate <= targetFPS && targetFPS <= $0.maxFrameRate
            }
            return dims.width == targetWidth &&
                   tenBitSubtypes.contains(subType) &&
                   supportsFPS &&
                   !format.isVideoBinned
        }

        var chosenFormat: AVCaptureDevice.Format?
        var useNativeAppleLog = false

        #if compiler(>=5.9)
        if #available(iOS 17.0, *) {
            if let nativeLogFormat = candidates.first(where: { $0.supportedColorSpaces.contains(.appleLog) }) {
                chosenFormat = nativeLogFormat
                useNativeAppleLog = true
            }
        }
        #endif

        if chosenFormat == nil {
            chosenFormat = candidates.first(where: {
                $0.supportedColorSpaces.contains(.HLG_BT2020) &&
                CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
            }) ?? candidates.first(where: {
                $0.supportedColorSpaces.contains(.HLG_BT2020)
            }) ?? candidates.first
        }

        guard let activeFormat = chosenFormat else {
            throw NSError(domain: "CineLog13Pro", code: -10, userInfo: [NSLocalizedDescriptionKey: "未找到支持的 10-bit 传感器格式"])
        }

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        device.activeFormat = activeFormat

        let frameDuration = CMTime(value: 1000, timescale: CMTimeScale(targetFPS * 1000))
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration

        #if compiler(>=5.9)
        if #available(iOS 17.0, *), useNativeAppleLog {
            device.activeColorSpace = .appleLog
            self.hardwareSupportsNativeAppleLog = true
        } else if activeFormat.supportedColorSpaces.contains(.HLG_BT2020) {
            device.activeColorSpace = .HLG_BT2020
            if activeFormat.isVideoHDRSupported {
                device.automaticallyAdjustsVideoHDREnabled = false
                device.isVideoHDREnabled = true
            }
            self.hardwareSupportsNativeAppleLog = false
        }
        #else
        if activeFormat.supportedColorSpaces.contains(.HLG_BT2020) {
            device.activeColorSpace = .HLG_BT2020
            if activeFormat.isVideoHDRSupported {
                device.automaticallyAdjustsVideoHDREnabled = false
                device.isVideoHDREnabled = true
            }
            self.hardwareSupportsNativeAppleLog = false
        }
        #endif

        if device.isGeometricDistortionCorrectionSupported {
            device.isGeometricDistortionCorrectionEnabled = false
        }

        let minISOVal = activeFormat.minISO
        let maxISOVal = min(activeFormat.maxISO, 3200.0)
        let currentISOVal = device.iso
        let dims = CMVideoFormatDescriptionGetDimensions(activeFormat.formatDescription)
        let badge = useNativeAppleLog
            ? "ISP Native Apple Log • \(dims.width)x\(dims.height) 10-Bit"
            : "13 Pro Max 10-Bit BT.2020 → Metal Apple Log • \(dims.width)x\(dims.height)"

        DispatchQueue.main.async {
            self.minISO = minISOVal
            self.maxISO = maxISOVal
            self.currentISO = currentISOVal
            self.activePipelineBadge = badge
            self.statusMessage = "10-Bit Log Pipeline Active"
        }
    }

    // MARK: - Cinema Exposure (180° Shutter Rule), ISO & Kelvin White Balance

    func updateFrameRateAndShutter() {
        sessionQueue.async { [weak self] in
            guard let self = self, let device = self.videoDevice else { return }
            do {
                try self.selectBest10BitLogFormat(for: device, targetWidth: 3840, targetFPS: self.selectedFPS.rawValue)
                try device.lockForConfiguration()
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
            } catch {
                DispatchQueue.main.async { self.statusMessage = "快门/帧率更新异常: \(error.localizedDescription)" }
            }
        }
    }

    func setManualISO(_ isoValue: Float) {
        currentISO = isoValue
        guard isManualExposure else { return }
        updateFrameRateAndShutter()
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

            let fileName = "CineLog13Pro_\(Int(Date().timeIntervalSince1970)).mov"
            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
            try? FileManager.default.removeItem(at: outputURL)

            do {
                let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)

                let colorProperties: [String: Any] = [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_2100_HLG,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
                ]

                var videoSettings: [String: Any] = [
                    AVVideoCodecKey: self.selectedCodec.avCodecType.rawValue,
                    AVVideoWidthKey: width,
                    AVVideoHeightKey: height,
                    AVVideoColorPropertiesKey: colorProperties
                ]

                if self.selectedCodec == .hevc10Bit {
                    videoSettings[AVVideoCompressionPropertiesKey] = [
                        AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel as String,
                        AVVideoAverageBitRateKey: 120_000_000,
                        AVVideoExpectedSourceFrameRateKey: self.selectedFPS.rawValue
                    ]
                }

                let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
                vInput.expectsMediaDataInRealTime = true

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

                DispatchQueue.main.async {
                    self.isRecording = true
                    self.recordingDuration = 0
                    self.statusMessage = "Recording \(self.selectedCodec.rawValue) (Apple Log)..."
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
                    self.statusMessage = "正在保存 Apple Log 视频至相册..."
                }
                if let url = outputURL {
                    self.saveVideoToPhotoLibrary(url: url)
                }
            }
        }
    }

    private func saveVideoToPhotoLibrary(url: URL) {
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
                        self.statusMessage = "已保存 Apple Log 视频 (\(self.selectedCodec.rawValue)) 至系统相册"
                    } else {
                        self.statusMessage = "相册保存失败: \(error?.localizedDescription ?? "未知错误")"
                    }
                }
            }
        }
    }

    // MARK: - Metal Buffer Pool Management

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

// MARK: - 4. AVCaptureVideoDataOutput & Audio Sample Buffer Delegate

extension AppleLogCameraEngine: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {

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
              let pipelineState = computePipelineState else {
            return
        }

        let width  = CVPixelBufferGetWidth(inputPixelBuffer)
        let height = CVPixelBufferGetHeight(inputPixelBuffer)
        ensurePixelBufferPools(width: Int32(width), height: Int32(height))

        let lumaW = CVPixelBufferGetWidthOfPlane(inputPixelBuffer, 0)
        let lumaH = CVPixelBufferGetHeightOfPlane(inputPixelBuffer, 0)
        let chromaW = CVPixelBufferGetWidthOfPlane(inputPixelBuffer, 1)
        let chromaH = CVPixelBufferGetHeightOfPlane(inputPixelBuffer, 1)

        var cvLumaTex: CVMetalTexture?
        var cvChromaTex: CVMetalTexture?

        let statusY = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, inputPixelBuffer, nil, .r16Unorm, lumaW, lumaH, 0, &cvLumaTex
        )
        let statusCbCr = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, inputPixelBuffer, nil, .rg16Unorm, chromaW, chromaH, 1, &cvChromaTex
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
            kCFAllocatorDefault, cache, logOutPB, nil, .rgba16Float, width, height, 0, &cvLogOutTex
        )
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, prevOutPB, nil, .rgba16Float, width, height, 0, &cvPrevOutTex
        )

        guard let logOutRef = cvLogOutTex, let prevOutRef = cvPrevOutTex,
              let logOutTex = CVMetalTextureGetTexture(logOutRef),
              let prevOutTex = CVMetalTextureGetTexture(prevOutRef),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        var uniforms = LogPipelineUniforms(
            isNativeAppleLog: hardwareSupportsNativeAppleLog ? 1 : 0,
            previewLUTMode: Int32(previewMode.rawValue),
            showZebraOverlay: isZebraEnabled ? 1 : 0,
            exposureBiasEV: exposureBiasEV,
            highlightRollOff: highlightRollOff,
            frameTime: Float(CACurrentMediaTime() - startTimestamp)
        )

        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(lumaTex, index: 0)
        encoder.setTexture(chromaTex, index: 1)
        encoder.setTexture(logOutTex, index: 2)
        encoder.setTexture(prevOutTex, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<LogPipelineUniforms>.stride, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (height + threadGroupSize.height - 1) / threadGroupSize.height,
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
        if frameCounter % 6 == 0 {
            computeFastLumaHistogram(from: inputPixelBuffer)
        }
    }

    private func computeFastLumaHistogram(from pixelBuffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddr = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uint16Ptr = baseAddr.assumingMemoryBound(to: UInt16.self)
        let strideU16 = bytesPerRow / 2

        var bins = Array(repeating: Float(0), count: 64)
        var totalSamples: Float = 0

        for y in stride(from: 0, to: height, by: 16) {
            let rowOffset = y * strideU16
            for x in stride(from: 0, to: width, by: 16) {
                let raw16 = uint16Ptr[rowOffset + x]
                let normalized = Float(raw16) / 65535.0
                let binIdx = min(63, max(0, Int(normalized * 63.0)))
                bins[binIdx] += 1
                totalSamples += 1
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
}

extension AppleLogCameraEngine {
    struct LogPipelineUniforms {
        var isNativeAppleLog: Int32
        var previewLUTMode: Int32
        var showZebraOverlay: Int32
        var exposureBiasEV: Float
        var highlightRollOff: Float
        var frameTime: Float
    }
}

// MARK: - 5. EDR 10-Bit Metal Viewfinder with Hardware Render Pipeline Scaling

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

// MARK: - 6. Cinema Pro HUD & SwiftUI Interface

struct LumaHistogramOverlay: View {
    let bins: [Float]

    var body: some View {
        HStack(alignment: .bottom, spacing: 1) {
            ForEach(0..<bins.count, id: \.self) { index in
                Rectangle()
                    .fill(index > 58 ? Color.red : (index >= 28 && index <= 34 ? Color.green : Color.white.opacity(0.85)))
                    .frame(width: 2, height: max(2, CGFloat(bins[index]) * 38))
            }
        }
        .padding(6)
        .background(Color.black.opacity(0.65))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.white.opacity(0.2), lineWidth: 1)
        )
    }
}

struct CineLogCameraView: View {
    @StateObject private var engine = AppleLogCameraEngine()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            MetalLogViewfinder(engine: engine)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(engine.activePipelineBadge)
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(.yellow)
                        Text(engine.statusMessage)
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundColor(.white.opacity(0.85))
                    }

                    Spacer()

                    LumaHistogramOverlay(bins: engine.lumaHistogram)
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                HStack(spacing: 8) {
                    Picker("Codec", selection: $engine.selectedCodec) {
                        ForEach(RecordingCodecOption.allCases) { codec in
                            Text(codec.rawValue).tag(codec)
                        }
                    }
                    .pickerStyle(.menu)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.black.opacity(0.65))
                    .foregroundColor(.white)
                    .cornerRadius(6)

                    Picker("FPS", selection: $engine.selectedFPS) {
                        ForEach(FrameRateOption.allCases) { fps in
                            Text(fps.label).tag(fps)
                        }
                    }
                    .pickerStyle(.menu)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.black.opacity(0.65))
                    .foregroundColor(.white)
                    .cornerRadius(6)

                    Picker("LUT", selection: $engine.previewMode) {
                        ForEach(PreviewLUTMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 240)

                    Button(action: { engine.isZebraEnabled.toggle() }) {
                        Text("ZEBRA")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(engine.isZebraEnabled ? Color.red : Color.black.opacity(0.65))
                            .foregroundColor(.white)
                            .cornerRadius(6)
                    }
                }
                .padding(.horizontal, 16)

                Spacer()

                VStack(spacing: 10) {
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("LOG EV BIAS: \(String(format: "%+.1f", engine.exposureBiasEV)) EV")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.white)
                            Slider(value: $engine.exposureBiasEV, in: -2.5...2.5, step: 0.1)
                                .tint(.yellow)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("HIGHLIGHT ROLL-OFF: \(Int(engine.highlightRollOff * 100))%")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.white)
                            Slider(value: $engine.highlightRollOff, in: 0.0...1.0, step: 0.05)
                                .tint(.orange)
                        }
                    }
                    .padding(.horizontal, 16)

                    HStack(spacing: 12) {
                        Toggle("MANUAL EXP", isOn: $engine.isManualExposure)
                            .labelsHidden()
                            .tint(.yellow)

                        Text(engine.isManualExposure ? "MANUAL" : "AUTO EXP")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(engine.isManualExposure ? .yellow : .white)

                        if engine.isManualExposure {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("ISO \(Int(engine.currentISO))")
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(.yellow)
                                Slider(
                                    value: Binding(
                                        get: { engine.currentISO },
                                        set: { engine.setManualISO($0) }
                                    ),
                                    in: engine.minISO...engine.maxISO
                                )
                                .tint(.yellow)
                            }

                            VStack(alignment: .leading, spacing: 2) {
                                Text("SHUTTER \(Int(engine.shutterAngleDegrees))°")
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(.yellow)
                                Slider(value: $engine.shutterAngleDegrees, in: 45...360, step: 15)
                                    .tint(.yellow)
                            }
                        }

                        Spacer()

                        Toggle("WB LOCK", isOn: $engine.isWhiteBalanceLocked)
                            .labelsHidden()
                            .tint(.cyan)
                        Text(engine.isWhiteBalanceLocked ? "\(Int(engine.whiteBalanceKelvin))K" : "AWB")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(engine.isWhiteBalanceLocked ? .cyan : .white)
                    }
                    .padding(.horizontal, 16)

                    HStack {
                        Text(formatTimecode(engine.recordingDuration, fps: engine.selectedFPS.rawValue))
                            .font(.system(size: 16, weight: .bold, design: .monospaced))
                            .foregroundColor(engine.isRecording ? .red : .white)
                            .frame(width: 130, alignment: .leading)

                        Spacer()

                        Button(action: { engine.toggleRecording() }) {
                            ZStack {
                                Circle()
                                    .stroke(Color.white, lineWidth: 3)
                                    .frame(width: 68, height: 68)
                                RoundedRectangle(cornerRadius: engine.isRecording ? 8 : 30)
                                    .fill(Color.red)
                                    .frame(
                                        width: engine.isRecording ? 30 : 54,
                                        height: engine.isRecording ? 30 : 54
                                    )
                                    .animation(.easeInOut(duration: 0.2), value: engine.isRecording)
                            }
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 2) {
                            Text("COLOR: BT.2020")
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .foregroundColor(.gray)
                            Text("CURVE: APPLE LOG")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundColor(.green)
                        }
                        .frame(width: 130, alignment: .trailing)
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 12)
                }
                .padding(.vertical, 10)
                .background(Color.black.opacity(0.72))
            }
        }
        .onAppear {
            engine.requestPermissionsAndStart()
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

// MARK: - 7. App Entry Point

@main
struct CineLog13ProApp: App {
    var body: some Scene {
        WindowGroup {
            CineLogCameraView()
                .preferredColorScheme(.dark)
        }
    }
}