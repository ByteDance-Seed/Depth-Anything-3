// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

// Depth Anything 3 – MLX-Swift Wrapper
//
// Complete inference wrapper supporting:
// - Multiple input images (dynamic batch of views)
// - Intrinsics/extrinsics conditioning
// - Dynamic spatial resolution (any multiple of 14)
// - Single forward pass → predicted depth per image
//
// Architecture:
//   DinoV2 backbone (ViT) → DualDPT depth head → depth maps
//   Optional CameraEnc for extrinsics/intrinsics conditioning

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Model configuration matching the Python YAML configs.
public struct DA3Config {
    public let backboneName: String
    public let embedDim: Int
    public let depth: Int
    public let numHeads: Int
    public let outLayers: [Int]
    public let altStart: Int
    public let qknormStart: Int
    public let ropeStart: Int
    public let catToken: Bool
    public let ffnLayer: String
    public let headDimIn: Int
    public let headOutputDim: Int
    public let headFeatures: Int
    public let headOutChannels: [Int]
    public let camEncDimOut: Int
    public let camDecDimIn: Int
    public let patchSize: Int

    public static let da3Small = DA3Config(
        backboneName: "vits", embedDim: 384, depth: 12, numHeads: 6,
        outLayers: [5, 7, 9, 11], altStart: 4, qknormStart: 4, ropeStart: 4,
        catToken: true, ffnLayer: "mlp",
        headDimIn: 768, headOutputDim: 2, headFeatures: 64,
        headOutChannels: [48, 96, 192, 384],
        camEncDimOut: 384, camDecDimIn: 768, patchSize: 14
    )

    public static let da3Base = DA3Config(
        backboneName: "vitb", embedDim: 768, depth: 12, numHeads: 12,
        outLayers: [5, 7, 9, 11], altStart: 4, qknormStart: 4, ropeStart: 4,
        catToken: true, ffnLayer: "mlp",
        headDimIn: 1536, headOutputDim: 2, headFeatures: 128,
        headOutChannels: [96, 192, 384, 768],
        camEncDimOut: 768, camDecDimIn: 1536, patchSize: 14
    )

    public static let da3Large = DA3Config(
        backboneName: "vitl", embedDim: 1024, depth: 24, numHeads: 16,
        outLayers: [11, 15, 19, 23], altStart: 8, qknormStart: 8, ropeStart: 8,
        catToken: true, ffnLayer: "mlp",
        headDimIn: 2048, headOutputDim: 2, headFeatures: 256,
        headOutChannels: [256, 512, 1024, 1024],
        camEncDimOut: 1024, camDecDimIn: 2048, patchSize: 14
    )

    public static let da3Giant = DA3Config(
        backboneName: "vitg", embedDim: 1536, depth: 40, numHeads: 24,
        outLayers: [19, 27, 33, 39], altStart: 13, qknormStart: 13, ropeStart: 13,
        catToken: true, ffnLayer: "swiglu",
        headDimIn: 3072, headOutputDim: 2, headFeatures: 256,
        headOutChannels: [256, 512, 1024, 1024],
        camEncDimOut: 1536, camDecDimIn: 3072, patchSize: 14
    )
}

// MARK: - Utility Functions

/// ImageNet normalization constants.
private let imageNetMean: [Float] = [0.485, 0.456, 0.406]
private let imageNetStd: [Float] = [0.229, 0.224, 0.225]

/// Bilinear interpolation for NHWC tensors.
func bilinearInterpolate(_ x: MLXArray, targetH: Int, targetW: Int, alignCorners: Bool = true)
    -> MLXArray
{
    let shape = x.shape
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3])
    if h == targetH && w == targetW { return x }

    let yCoords: MLXArray
    let xCoords: MLXArray
    if alignCorners && targetH > 1 && targetW > 1 {
        yCoords = MLXArray.linspace(Float(0), Float(h - 1), count: targetH)
        xCoords = MLXArray.linspace(Float(0), Float(w - 1), count: targetW)
    } else {
        yCoords = clip(
            (MLXArray(Float32(0) ..< Float32(targetH)) + 0.5) * Float(h) / Float(targetH) - 0.5,
            min: 0,
            max: Float(h - 1))
        xCoords = clip(
            (MLXArray(Float32(0) ..< Float32(targetW)) + 0.5) * Float(w) / Float(targetW) - 0.5,
            min: 0,
            max: Float(w - 1))
    }

    let y0 = clip(floor(yCoords).asType(.int32), min: 0, max: h - 1)
    let y1 = clip(y0 + 1, min: 0, max: h - 1)
    let x0 = clip(floor(xCoords).asType(.int32), min: 0, max: w - 1)
    let x1 = clip(x0 + 1, min: 0, max: w - 1)

    let wy = expandedDimensions(yCoords - y0.asType(.float32), axes: [0, 2, 3])
    let wx = expandedDimensions(xCoords - x0.asType(.float32), axes: [0, 1, 3])

    let topLeft = x[0..., y0, x0]
    let topRight = x[0..., y0, x1]
    let botLeft = x[0..., y1, x0]
    let botRight = x[0..., y1, x1]

    return topLeft * (1 - wy) * (1 - wx) + topRight * (1 - wy) * wx + botLeft * wy * (1 - wx)
        + botRight * wy * wx
}

/// Create normalized UV grid for positional embeddings.
func createUVGrid(width: Int, height: Int, aspectRatio: Float? = nil) -> MLXArray {
    let ar = aspectRatio ?? Float(width) / Float(height)
    let diag = sqrt(ar * ar + 1.0)
    let sx = ar / diag
    let sy: Float = 1.0 / diag
    let lx = -sx * Float(width - 1) / Float(width)
    let rx = sx * Float(width - 1) / Float(width)
    let ty = -sy * Float(height - 1) / Float(height)
    let by = sy * Float(height - 1) / Float(height)
    let xs = MLXArray.linspace(lx, rx, count: width)
    let ys = MLXArray.linspace(ty, by, count: height)
    let xx = broadcastTo(expandedDimensions(xs, axis: 0), shape: [height, width])
    let yy = broadcastTo(expandedDimensions(ys, axis: 1), shape: [height, width])
    return stacked([xx, yy], axis: -1)
}

/// Sinusoidal positional embedding.
func makeSincosPosEmbed(embedDim: Int, pos: MLXArray, omega0: Float = 100.0) -> MLXArray {
    let omega = MLXArray(Float32(0) ..< Float32(embedDim / 2)) / Float(embedDim / 2)
    let invFreq = 1.0 / pow(MLXArray(omega0), omega)
    let posFlat = pos.reshaped([-1]).asType(.float32)
    let outer = expandedDimensions(posFlat, axis: 1) * expandedDimensions(invFreq, axis: 0)
    return concatenated([sin(outer), cos(outer)], axis: 1).asType(.float32)
}

/// Position grid to sinusoidal embed.
func positionGridToEmbed(_ grid: MLXArray, embedDim: Int, omega0: Float = 100.0) -> MLXArray {
    let shape = grid.shape
    let (h, w) = (shape[0], shape[1])
    let flat = grid.reshaped([-1, 2])
    let embX = makeSincosPosEmbed(embedDim: embedDim / 2, pos: flat[0..., 0], omega0: omega0)
    let embY = makeSincosPosEmbed(embedDim: embedDim / 2, pos: flat[0..., 1], omega0: omega0)
    return concatenated([embX, embY], axis: -1).reshaped([h, w, embedDim])
}

/// Affine inverse of a rigid transformation matrix.
func affineInverse(_ a: MLXArray) -> MLXArray {
    let r = a[0..., 0..<3, 0..<3]
    let t = a[0..., 0..<3, 3...]
    let p = a[0..., 3..., 0...]
    let rt = r.transposed(0, 1, 3, 2)
    return concatenated([concatenated([rt, -matmul(rt, t)], axis: -1), p], axis: -2)
}

/// Quaternion (XYZW) to rotation matrix.
func quatToMat(_ q: MLXArray) -> MLXArray {
    let i = q[0..., 0]
    let j = q[0..., 1]
    let k = q[0..., 2]
    let r = q[0..., 3]
    let twoS = 2.0 / sum(q * q, axis: -1)
    let elements = stacked([
        1 - twoS * (j * j + k * k),
        twoS * (i * j - k * r),
        twoS * (i * k + j * r),
        twoS * (i * j + k * r),
        1 - twoS * (i * i + k * k),
        twoS * (j * k - i * r),
        twoS * (i * k - j * r),
        twoS * (j * k + i * r),
        1 - twoS * (i * i + j * j),
    ], axis: -1)
    var outShape = Array(q.shape.dropLast())
    outShape.append(contentsOf: [3, 3])
    return elements.reshaped(outShape)
}

/// Rotation matrix to quaternion (XYZW).
func matToQuat(_ matrix: MLXArray) -> MLXArray {
    let batchDim = Array(matrix.shape.dropLast(2))
    let flat = matrix.reshaped(batchDim + [9])
    let m00 = flat[0..., 0]
    let m01 = flat[0..., 1]
    let m02 = flat[0..., 2]
    let m10 = flat[0..., 3]
    let m11 = flat[0..., 4]
    let m12 = flat[0..., 5]
    let m20 = flat[0..., 6]
    let m21 = flat[0..., 7]
    let m22 = flat[0..., 8]

    let qAbs = sqrt(
        maximum(
            stacked([
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ], axis: -1), 0))

    let quatByRijk = stacked([
        stacked([qAbs[0..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis: -1),
        stacked([m21 - m12, qAbs[0..., 1] ** 2, m10 + m01, m02 + m20], axis: -1),
        stacked([m02 - m20, m10 + m01, qAbs[0..., 2] ** 2, m12 + m21], axis: -1),
        stacked([m10 - m01, m20 + m02, m21 + m12, qAbs[0..., 3] ** 2], axis: -1),
    ], axis: -2)

    let denom = 2.0 * maximum(expandedDimensions(qAbs, axis: -1), MLXArray(Float(0.1)))
    let candidates = quatByRijk / denom
    let bestIdx = argMax(qAbs, axis: -1)
    let oneHot = MLX.oneHot(bestIdx, num: 4)
    let mask = expandedDimensions(oneHot, axis: -1)
    var out = sum(candidates * mask, axis: -2)
    // [r, i, j, k] -> [i, j, k, r]
    out = concatenated([out[0..., 1..<2], out[0..., 2..<3], out[0..., 3..<4], out[0..., 0..<1]], axis: -1)
    // Standardize: make real part non-negative
    return MLX.where(out[0..., 3...] .< 0, -out, out)
}

/// Convert extrinsics+intrinsics to 9D pose encoding.
func extriIntriToPoseEncoding(extrinsics: MLXArray, intrinsics: MLXArray, imageSize: (Int, Int))
    -> MLXArray
{
    let r = extrinsics[0..., 0..., 0..<3, 0..<3]
    let t = extrinsics[0..., 0..., 0..<3, 3]
    let quat = matToQuat(r)
    let (h, w) = imageSize
    let fovH = 2 * atan(MLXArray(Float(h) / 2.0) / intrinsics[0..., 0..., 1, 1])
    let fovW = 2 * atan(MLXArray(Float(w) / 2.0) / intrinsics[0..., 0..., 0, 0])
    return concatenated(
        [t, quat, expandedDimensions(fovH, axis: -1), expandedDimensions(fovW, axis: -1)], axis: -1
    ).asType(.float32)
}

// MARK: - Layer Scale

class LayerScaleLayer: Module {
    let gamma: MLXArray

    init(dim: Int, initValues: Float = 1e-5) {
        gamma = MLXArray.ones([dim]) * initValues
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

// MARK: - MLP

class MlpLayer: Module {
    let fc1: Linear
    let fc2: Linear

    init(inFeatures: Int, hiddenFeatures: Int, outFeatures: Int, bias: Bool = true) {
        fc1 = Linear(inFeatures, hiddenFeatures, bias: bias)
        fc2 = Linear(hiddenFeatures, outFeatures, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = gelu(h)
        h = fc2(h)
        return h
    }
}

// MARK: - SwiGLU FFN

class SwiGLUFFNLayer: Module {
    let w12: Linear
    let w3: Linear

    init(inFeatures: Int, hiddenFeatures: Int, outFeatures: Int, bias: Bool = true) {
        let hf = ((hiddenFeatures * 2 / 3) + 7) / 8 * 8
        w12 = Linear(inFeatures, 2 * hf, bias: bias)
        w3 = Linear(hf, outFeatures, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x12 = w12(x)
        let half = x12.shape.last! / 2
        let x1 = x12[0..., 0..<half]
        let x2 = x12[0..., half...]
        let hidden = silu(x1) * x2
        return w3(hidden)
    }
}

// MARK: - 2D RoPE

class RotaryPositionEmbedding2DLayer: Module {
    let baseFrequency: Float

    init(frequency: Float = 100.0) {
        baseFrequency = frequency
    }

    private func computeFreq(dim: Int, seqLen: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let exponents = MLXArray(Float32(0) ..< Float32(dim)).asType(.float32)
            / Float(dim)
        let invFreq = 1.0 / pow(MLXArray(baseFrequency), exponents[stride: 2])
        let positions = MLXArray(Float32(0) ..< Float32(seqLen))
        let angles = expandedDimensions(positions, axis: 1) * expandedDimensions(invFreq, axis: 0)
        let anglesDouble = concatenated([angles, angles], axis: -1).asType(dtype)
        return (cos(anglesDouble), sin(anglesDouble))
    }

    private func rotateFeatures(_ x: MLXArray) -> MLXArray {
        let half = x.shape.last! / 2
        let x1 = x[0..., 0..<half]
        let x2 = x[0..., half...]
        return concatenated([-x2, x1], axis: -1)
    }

    private func apply1D(_ tokens: MLXArray, positions: MLXArray, cosComp: MLXArray, sinComp: MLXArray) -> MLXArray {
        let cosEmb = expandedDimensions(cosComp[positions], axis: 1)
        let sinEmb = expandedDimensions(sinComp[positions], axis: 1)
        return tokens * cosEmb + rotateFeatures(tokens) * sinEmb
    }

    func callAsFunction(tokens: MLXArray, positions: MLXArray) -> MLXArray {
        let featDim = tokens.shape.last! / 2
        let maxPos = Int(positions.max().item(Int.self)) + 1
        let (cosComp, sinComp) = computeFreq(dim: featDim, seqLen: maxPos, dtype: tokens.dtype)
        let vert = tokens[0..., 0..<featDim]
        let horiz = tokens[0..., featDim...]
        let posY = positions[0..., 0].asType(.int32)
        let posX = positions[0..., 1].asType(.int32)
        let vertOut = apply1D(vert, positions: posY, cosComp: cosComp, sinComp: sinComp)
        let horizOut = apply1D(horiz, positions: posX, cosComp: cosComp, sinComp: sinComp)
        return concatenated([vertOut, horizOut], axis: -1)
    }
}

// MARK: - Attention

class AttentionLayer: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let qkv: Linear
    let qNorm: LayerNorm?
    let kNorm: LayerNorm?
    let proj: Linear
    let rope: RotaryPositionEmbedding2DLayer?

    init(dim: Int, numHeads: Int, qkvBias: Bool = false, projBias: Bool = true,
         qkNorm: Bool = false, rope: RotaryPositionEmbedding2DLayer? = nil)
    {
        self.numHeads = numHeads
        headDim = dim / numHeads
        scale = pow(Float(headDim), -0.5)
        qkv = Linear(dim, dim * 3, bias: qkvBias)
        qNorm = qkNorm ? LayerNorm(dimensions: headDim) : nil
        kNorm = qkNorm ? LayerNorm(dimensions: headDim) : nil
        proj = Linear(dim, dim, bias: projBias)
        self.rope = rope
    }

    func callAsFunction(_ x: MLXArray, pos: MLXArray? = nil, attnMask: MLXArray? = nil) -> MLXArray
    {
        let (b, n, c) = (x.shape[0], x.shape[1], x.shape[2])
        var qkvOut = qkv(x).reshaped([b, n, 3, numHeads, headDim])
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)  // (3, B, heads, N, headDim)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        if let qn = qNorm { q = qn(q) }
        if let kn = kNorm { k = kn(k) }
        if let r = rope, let p = pos {
            q = r.callAsFunction(tokens: q, positions: p)
            k = r.callAsFunction(tokens: k, positions: p)
        }
        q = q * scale
        var attn = matmul(q, k.transposed(0, 1, 3, 2))
        if let mask = attnMask {
            let expanded = broadcastTo(expandedDimensions(mask, axis: 1), shape: attn.shape)
            attn = attn + MLX.where(expanded, MLXArray(0.0), MLXArray(Float(-1e9)))
        }
        attn = softmax(attn, axis: -1)
        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, c])
        return proj(out)
    }
}

// MARK: - Transformer Block

class TransformerBlock: Module {
    let norm1: LayerNorm
    let attn: AttentionLayer
    let ls1: LayerScaleLayer?
    let norm2: LayerNorm
    let mlp: Module
    let ls2: LayerScaleLayer?

    init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, qkvBias: Bool = false,
         projBias: Bool = true, ffnBias: Bool = true, initValues: Float? = nil,
         ffnLayer: String = "mlp", qkNorm: Bool = false,
         rope: RotaryPositionEmbedding2DLayer? = nil)
    {
        norm1 = LayerNorm(dimensions: dim, eps: 1e-6)
        attn = AttentionLayer(
            dim: dim, numHeads: numHeads, qkvBias: qkvBias,
            projBias: projBias, qkNorm: qkNorm, rope: rope)
        ls1 = initValues != nil ? LayerScaleLayer(dim: dim, initValues: initValues!) : nil
        norm2 = LayerNorm(dimensions: dim, eps: 1e-6)
        let mlpHidden = Int(Float(dim) * mlpRatio)
        if ffnLayer == "swiglu" {
            mlp = SwiGLUFFNLayer(
                inFeatures: dim, hiddenFeatures: mlpHidden, outFeatures: dim, bias: ffnBias)
        } else {
            mlp = MlpLayer(
                inFeatures: dim, hiddenFeatures: mlpHidden, outFeatures: dim, bias: ffnBias)
        }
        ls2 = initValues != nil ? LayerScaleLayer(dim: dim, initValues: initValues!) : nil
    }

    func callAsFunction(_ x: MLXArray, pos: MLXArray? = nil, attnMask: MLXArray? = nil)
        -> MLXArray
    {
        var attnOut = attn(norm1(x), pos: pos, attnMask: attnMask)
        if let l = ls1 { attnOut = l(attnOut) }
        var h = x + attnOut
        var ffnOut = (mlp as! any UnaryLayer).callAsFunction(norm2(h))
        if let l = ls2 { ffnOut = l(ffnOut) }
        h = h + ffnOut
        return h
    }
}

// MARK: - Patch Embedding

class PatchEmbedLayer: Module {
    let patchSize: Int
    let numPatches: Int
    let proj: Conv2d

    init(imgSize: Int = 518, patchSize: Int = 14, inChans: Int = 3, embedDim: Int = 768) {
        self.patchSize = patchSize
        numPatches = (imgSize / patchSize) * (imgSize / patchSize)
        proj = Conv2d(
            inputChannels: inChans, outputChannels: embedDim,
            kernelSize: .init(patchSize, patchSize),
            stride: .init(patchSize, patchSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) NHWC
        let out = proj(x)  // (B, H/p, W/p, D)
        let b = out.shape[0]
        return out.reshaped([b, -1, out.shape[3]])  // (B, N, D)
    }
}

// MARK: - ResidualConvUnit

class ResidualConvUnitLayer: Module {
    let conv1: Conv2d
    let conv2: Conv2d

    init(features: Int) {
        conv1 = Conv2d(
            inputChannels: features, outputChannels: features,
            kernelSize: .init(3, 3), stride: .init(1, 1), padding: .init(1, 1))
        conv2 = Conv2d(
            inputChannels: features, outputChannels: features,
            kernelSize: .init(3, 3), stride: .init(1, 1), padding: .init(1, 1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(x)
        out = conv1(out)
        out = relu(out)
        out = conv2(out)
        return out + x
    }
}

// MARK: - FeatureFusionBlock

class FeatureFusionBlockLayer: Module {
    let hasResidual: Bool
    let resConfUnit1: ResidualConvUnitLayer?
    let resConfUnit2: ResidualConvUnitLayer
    let outConv: Conv2d

    init(features: Int, hasResidual: Bool = true) {
        self.hasResidual = hasResidual
        resConfUnit1 = hasResidual ? ResidualConvUnitLayer(features: features) : nil
        resConfUnit2 = ResidualConvUnitLayer(features: features)
        outConv = Conv2d(
            inputChannels: features, outputChannels: features,
            kernelSize: .init(1, 1))
    }

    func callAsFunction(_ top: MLXArray, lateral: MLXArray? = nil, size: (Int, Int)? = nil)
        -> MLXArray
    {
        var y = top
        if hasResidual, let lat = lateral, let rcu1 = resConfUnit1 {
            y = y + rcu1(lat)
        }
        y = resConfUnit2(y)
        let targetH = size?.0 ?? (y.shape[1] * 2)
        let targetW = size?.1 ?? (y.shape[2] * 2)
        y = bilinearInterpolate(y, targetH: targetH, targetW: targetW, alignCorners: true)
        y = outConv(y)
        return y
    }
}

// MARK: - Camera Encoder Block

class CamAttentionLayer: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let qkv: Linear
    let qNorm: LayerNorm?
    let kNorm: LayerNorm?
    let proj: Linear

    init(dim: Int, numHeads: Int, qkNorm: Bool = false) {
        self.numHeads = numHeads
        headDim = dim / numHeads
        scale = pow(Float(headDim), -0.5)
        qkv = Linear(dim, dim * 3, bias: true)
        qNorm = qkNorm ? LayerNorm(dimensions: headDim) : nil
        kNorm = qkNorm ? LayerNorm(dimensions: headDim) : nil
        proj = Linear(dim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n, c) = (x.shape[0], x.shape[1], x.shape[2])
        var qkvOut = qkv(x).reshaped([b, n, 3, numHeads, headDim])
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)
        var q = qkvOut[0]
        var k = qkvOut[1]
        let v = qkvOut[2]
        if let qn = qNorm { q = qn(q) }
        if let kn = kNorm { k = kn(k) }
        q = q * scale
        var attn = matmul(q, k.transposed(0, 1, 3, 2))
        attn = softmax(attn, axis: -1)
        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, c])
        return proj(out)
    }
}

class CamBlockLayer: Module {
    let norm1: LayerNorm
    let attn: CamAttentionLayer
    let ls1: LayerScaleLayer?
    let norm2: LayerNorm
    let mlp: MlpLayer
    let ls2: LayerScaleLayer?

    init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, initValues: Float? = nil) {
        norm1 = LayerNorm(dimensions: dim)
        attn = CamAttentionLayer(dim: dim, numHeads: numHeads)
        ls1 = initValues != nil ? LayerScaleLayer(dim: dim, initValues: initValues!) : nil
        norm2 = LayerNorm(dimensions: dim)
        let hidden = Int(Float(dim) * mlpRatio)
        mlp = MlpLayer(inFeatures: dim, hiddenFeatures: hidden, outFeatures: dim)
        ls2 = initValues != nil ? LayerScaleLayer(dim: dim, initValues: initValues!) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var attnOut = attn(norm1(x))
        if let l = ls1 { attnOut = l(attnOut) }
        var h = x + attnOut
        var ffnOut = mlp(norm2(h))
        if let l = ls2 { ffnOut = l(ffnOut) }
        h = h + ffnOut
        return h
    }
}

// MARK: - Camera Encoder

class CameraEncoder: Module {
    let poseBranch: MlpLayer
    let tokenNorm: LayerNorm
    let trunk: [CamBlockLayer]
    let trunkNorm: LayerNorm

    init(dimOut: Int = 1024, dimIn: Int = 9, trunkDepth: Int = 4,
         numHeads: Int = 16, mlpRatio: Int = 4, initValues: Float = 0.01)
    {
        poseBranch = MlpLayer(
            inFeatures: dimIn, hiddenFeatures: dimOut / 2, outFeatures: dimOut)
        tokenNorm = LayerNorm(dimensions: dimOut)
        trunk = (0..<trunkDepth).map { _ in
            CamBlockLayer(
                dim: dimOut, numHeads: numHeads, mlpRatio: Float(mlpRatio),
                initValues: initValues)
        }
        trunkNorm = LayerNorm(dimensions: dimOut)
    }

    func callAsFunction(ext: MLXArray, ixt: MLXArray, imageSize: (Int, Int)) -> MLXArray {
        let c2ws = affineInverse(ext)
        let poseEncoding = extriIntriToPoseEncoding(
            extrinsics: c2ws, intrinsics: ixt, imageSize: imageSize)
        var tokens = poseBranch(poseEncoding)
        tokens = tokenNorm(tokens)
        for blk in trunk { tokens = blk(tokens) }
        tokens = trunkNorm(tokens)
        return tokens
    }
}

// MARK: - DualDPT Depth Head

class DualDPTHead: Module {
    let patchSize: Int
    let activation: String
    let confActivation: String
    let posEmbed: Bool
    let downRatio: Int

    let norm: LayerNorm
    let projects: [Conv2d]
    let resize0: ConvTranspose2d
    let resize1: ConvTranspose2d
    let resize3: Conv2d
    let layer1Rn: Conv2d
    let layer2Rn: Conv2d
    let layer3Rn: Conv2d
    let layer4Rn: Conv2d
    let refinenet4: FeatureFusionBlockLayer
    let refinenet3: FeatureFusionBlockLayer
    let refinenet2: FeatureFusionBlockLayer
    let refinenet1: FeatureFusionBlockLayer
    let outputConv1: Conv2d
    let outputConv2a: Conv2d
    let outputConv2b: Conv2d

    init(dimIn: Int, patchSize: Int = 14, outputDim: Int = 2,
         activation: String = "exp", confActivation: String = "expp1",
         features: Int = 256, outChannels: [Int] = [256, 512, 1024, 1024],
         posEmbed: Bool = true, downRatio: Int = 1)
    {
        self.patchSize = patchSize
        self.activation = activation
        self.confActivation = confActivation
        self.posEmbed = posEmbed
        self.downRatio = downRatio

        norm = LayerNorm(dimensions: dimIn)
        projects = outChannels.map { oc in
            Conv2d(inputChannels: dimIn, outputChannels: oc, kernelSize: .init(1, 1))
        }
        resize0 = ConvTranspose2d(
            inputChannels: outChannels[0], outputChannels: outChannels[0],
            kernelSize: .init(4, 4), stride: .init(4, 4))
        resize1 = ConvTranspose2d(
            inputChannels: outChannels[1], outputChannels: outChannels[1],
            kernelSize: .init(2, 2), stride: .init(2, 2))
        resize3 = Conv2d(
            inputChannels: outChannels[3], outputChannels: outChannels[3],
            kernelSize: .init(3, 3), stride: .init(2, 2), padding: .init(1, 1))
        layer1Rn = Conv2d(
            inputChannels: outChannels[0], outputChannels: features,
            kernelSize: .init(3, 3), padding: .init(1, 1), bias: false)
        layer2Rn = Conv2d(
            inputChannels: outChannels[1], outputChannels: features,
            kernelSize: .init(3, 3), padding: .init(1, 1), bias: false)
        layer3Rn = Conv2d(
            inputChannels: outChannels[2], outputChannels: features,
            kernelSize: .init(3, 3), padding: .init(1, 1), bias: false)
        layer4Rn = Conv2d(
            inputChannels: outChannels[3], outputChannels: features,
            kernelSize: .init(3, 3), padding: .init(1, 1), bias: false)
        refinenet4 = FeatureFusionBlockLayer(features: features, hasResidual: false)
        refinenet3 = FeatureFusionBlockLayer(features: features, hasResidual: true)
        refinenet2 = FeatureFusionBlockLayer(features: features, hasResidual: true)
        refinenet1 = FeatureFusionBlockLayer(features: features, hasResidual: true)
        outputConv1 = Conv2d(
            inputChannels: features, outputChannels: features / 2,
            kernelSize: .init(3, 3), padding: .init(1, 1))
        outputConv2a = Conv2d(
            inputChannels: features / 2, outputChannels: 32,
            kernelSize: .init(3, 3), padding: .init(1, 1))
        outputConv2b = Conv2d(
            inputChannels: 32, outputChannels: outputDim,
            kernelSize: .init(1, 1))
    }

    func callAsFunction(feats: [(MLXArray, MLXArray)], h: Int, w: Int) -> (
        depth: MLXArray, depthConf: MLXArray
    ) {
        let (b, s, n, c) = (
            feats[0].0.shape[0], feats[0].0.shape[1],
            feats[0].0.shape[2], feats[0].0.shape[3]
        )
        let flatFeats = feats.map { $0.0.reshaped([b * s, n, c]) }
        let result = forwardImpl(feats: flatFeats, h: h, w: w, patchStartIdx: 0)
        let depth = result.depth.reshaped([b, s] + Array(result.depth.shape.dropFirst()))
        let conf = result.conf.reshaped([b, s] + Array(result.conf.shape.dropFirst()))
        return (depth, conf)
    }

    private func forwardImpl(feats: [MLXArray], h: Int, w: Int, patchStartIdx: Int) -> (
        depth: MLXArray, conf: MLXArray
    ) {
        let bs = feats[0].shape[0]
        let c = feats[0].shape[2]
        let ph = h / patchSize
        let pw = w / patchSize

        var resizedFeats = [MLXArray]()
        for stageIdx in 0..<4 {
            var x = feats[stageIdx][0..., patchStartIdx...]
            x = norm(x)
            x = x.reshaped([bs, ph, pw, c])
            x = projects[stageIdx](x)
            if posEmbed { x = addPosEmbed(x, w: w, h: h) }
            switch stageIdx {
            case 0: x = resize0(x)
            case 1: x = resize1(x)
            case 3: x = resize3(x)
            default: break
            }
            resizedFeats.append(x)
        }

        // Fuse
        let l1Rn = layer1Rn(resizedFeats[0])
        let l2Rn = layer2Rn(resizedFeats[1])
        let l3Rn = layer3Rn(resizedFeats[2])
        let l4Rn = layer4Rn(resizedFeats[3])

        var out = refinenet4(l4Rn, size: (l3Rn.shape[1], l3Rn.shape[2]))
        out = refinenet3(out, lateral: l3Rn, size: (l2Rn.shape[1], l2Rn.shape[2]))
        out = refinenet2(out, lateral: l2Rn, size: (l1Rn.shape[1], l1Rn.shape[2]))
        out = refinenet1(out, lateral: l1Rn)
        out = outputConv1(out)

        let hOut = ph * patchSize / downRatio
        let wOut = pw * patchSize / downRatio
        out = bilinearInterpolate(out, targetH: hOut, targetW: wOut, alignCorners: true)
        if posEmbed { out = addPosEmbed(out, w: w, h: h) }

        out = outputConv2a(out)
        out = relu(out)
        let logits = outputConv2b(out)

        let depthLogits = logits[0..., 0..., 0..., 0..<(logits.shape[3] - 1)]
        let confLogits = logits[0..., 0..., 0..., (logits.shape[3] - 1)...]
        let depth = applyActivation(depthLogits.squeezed(axis: -1), activation)
        let conf = applyActivation(confLogits.squeezed(axis: -1), confActivation)
        return (depth, conf)
    }

    private func addPosEmbed(_ x: MLXArray, w: Int, h: Int, ratio: Float = 0.1) -> MLXArray {
        let (ph, pw, c) = (x.shape[1], x.shape[2], x.shape[3])
        let pe = positionGridToEmbed(
            createUVGrid(width: pw, height: ph, aspectRatio: Float(w) / Float(h)),
            embedDim: c, omega0: 100.0) * ratio
        return x + expandedDimensions(pe, axis: 0)
    }

    private func applyActivation(_ x: MLXArray, _ act: String) -> MLXArray {
        switch act.lowercased() {
        case "exp": return exp(x)
        case "expp1": return exp(x) + 1
        case "relu": return relu(x)
        case "sigmoid": return sigmoid(x)
        case "softplus": return log(1 + exp(x))
        case "tanh": return tanh(x)
        default: return x
        }
    }
}

// MARK: - DinoV2 Backbone

class DinoV2Backbone: Module {
    let embedDim: Int
    let altStart: Int
    let ropeStart: Int
    let catToken: Bool
    let patchSize: Int
    let numRegisterTokens: Int
    let interpolateOffset: Float

    let patchEmbed: PatchEmbedLayer
    var clsToken: MLXArray
    var cameraToken: MLXArray?
    var posEmbed: MLXArray
    var registerTokens: MLXArray?
    let rope: RotaryPositionEmbedding2DLayer?
    let blocks: [TransformerBlock]
    let norm: LayerNorm
    let outLayers: [Int]

    init(config: DA3Config) {
        embedDim = config.embedDim
        altStart = config.altStart
        ropeStart = config.ropeStart
        catToken = config.catToken
        patchSize = config.patchSize
        numRegisterTokens = 0
        interpolateOffset = 0.1
        outLayers = config.outLayers

        patchEmbed = PatchEmbedLayer(
            imgSize: 518, patchSize: patchSize, inChans: 3, embedDim: embedDim)
        clsToken = MLXArray.zeros([1, 1, embedDim])
        cameraToken =
            altStart != -1 ? MLX.random.normal([1, 2, embedDim]) * 0.02 : nil
        posEmbed = MLXArray.zeros([1, patchEmbed.numPatches + 1, embedDim])
        registerTokens = nil

        rope =
            ropeStart != -1 ? RotaryPositionEmbedding2DLayer(frequency: 100.0) : nil

        blocks = (0..<config.depth).map { i in
            TransformerBlock(
                dim: embedDim, numHeads: config.numHeads, mlpRatio: 4.0,
                qkvBias: true, projBias: true, ffnBias: true,
                initValues: 1.0, ffnLayer: config.ffnLayer,
                qkNorm: config.qknormStart != -1 && i >= config.qknormStart,
                rope: (config.ropeStart != -1 && i >= config.ropeStart) ? rope : nil)
        }
        norm = LayerNorm(dimensions: embedDim)
    }

    func callAsFunction(_ x: MLXArray, camToken: MLXArray? = nil,
                        refViewStrategy: String = "saddle_balanced")
        -> [(MLXArray, MLXArray)]
    {
        let (b, s, h, w, _) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        // Prepare tokens
        let xFlat = x.reshaped([b * s, h, w, x.shape[4]])
        var patches = patchEmbed(xFlat)
        let cls = broadcastTo(clsToken, shape: [b * s, 1, embedDim])
        patches = concatenated([cls, patches], axis: 1)
        // Position embedding (skip interpolation for standard sizes)
        patches = patches + posEmbed
        var xTok = patches.reshaped([b, s, patches.shape[1], embedDim])

        // Prepare RoPE positions
        let ph = h / patchSize
        let pw = w / patchSize
        var pos: MLXArray?
        var posNodiff: MLXArray?
        if rope != nil {
            let yCoords = MLXArray(Int32(0) ..< Int32(ph))
            let xCoords = MLXArray(Int32(0) ..< Int32(pw))
            let yy = MLX.repeated(yCoords, count: pw)
            let xx = tiled(xCoords, repetitions: [ph])
            var rawPos = stacked([yy, xx], axis: -1).reshaped([1, ph * pw, 2]).asType(.float32)
            rawPos = broadcastTo(rawPos, shape: [b * s, ph * pw, 2]).reshaped([b, s, ph * pw, 2])
            rawPos = rawPos + 1  // offset for special token
            let special = MLXArray.zeros([b, s, 1, 2])
            pos = concatenated([special, rawPos], axis: 2)
            posNodiff = MLXArray.zeros(pos!.shape)
            posNodiff = posNodiff! + 1
            let specialNd = MLXArray.zeros([b, s, 1, 2])
            posNodiff = concatenated([specialNd, posNodiff![0..., 0..., 1..., 0...]], axis: 2)
        }

        var localX = xTok
        var outputs = [(MLXArray, MLXArray)]()
        let outSet = Set(outLayers)

        for (i, blk) in blocks.enumerated() {
            let gPos: MLXArray? = (i >= ropeStart && rope != nil) ? posNodiff : nil
            let lPos: MLXArray? = (i >= ropeStart && rope != nil) ? pos : nil

            // Camera token injection
            if altStart != -1 && i == altStart {
                if let ct = camToken {
                    // Replace cls token with camera token
                    for bidx in 0..<b {
                        for sidx in 0..<s {
                            xTok[bidx, sidx, 0] = ct[bidx, sidx]
                        }
                    }
                } else if let ct = cameraToken {
                    let ref = broadcastTo(ct[0..., 0..<1], shape: [b, 1, embedDim])
                    let src = broadcastTo(ct[0..., 1...], shape: [b, s - 1, embedDim])
                    let camTok = concatenated([ref, src], axis: 1)
                    for bidx in 0..<b {
                        for sidx in 0..<s {
                            xTok[bidx, sidx, 0] = camTok[bidx, sidx]
                        }
                    }
                }
            }

            // Local or global attention
            if altStart != -1 && i >= altStart && i % 2 == 1 {
                // Global attention
                let flat = xTok.reshaped([b, s * xTok.shape[2], xTok.shape[3]])
                let pFlat = gPos?.reshaped([b, s * gPos!.shape[2], gPos!.shape[3]])
                let out = blk(flat, pos: pFlat)
                xTok = out.reshaped([b, s, xTok.shape[2], xTok.shape[3]])
            } else {
                // Local attention
                let flat = xTok.reshaped([b * s, xTok.shape[2], xTok.shape[3]])
                let pFlat = lPos?.reshaped([b * s, lPos!.shape[2], lPos!.shape[3]])
                let out = blk(flat, pos: pFlat)
                xTok = out.reshaped([b, s, xTok.shape[2], xTok.shape[3]])
                localX = xTok
            }

            if outSet.contains(i) {
                let outX: MLXArray
                if catToken {
                    outX = concatenated([localX, xTok], axis: -1)
                } else {
                    outX = xTok
                }
                let camTokOut = outX[0..., 0..., 0]  // (B, S, D)
                outputs.append((outX, camTokOut))
            }
        }

        // Apply final norm and drop cls/register tokens
        let skip = 1 + numRegisterTokens
        return outputs.map { (outX, camTokOut) in
            let normed: MLXArray
            if outX.shape.last! == embedDim {
                normed = norm(outX)
            } else {
                normed = concatenated(
                    [outX[0..., 0..., 0..., 0..<embedDim],
                     norm(outX[0..., 0..., 0..., embedDim...])],
                    axis: -1)
            }
            return (normed[0..., 0..., skip..., 0...], camTokOut)
        }
    }
}

// MARK: - Main Model

/// Depth Anything 3 model for MLX-Swift.
///
/// Supports multiple input images, extrinsics/intrinsics conditioning,
/// dynamic spatial resolution, and returns predicted depth per image.
public class DepthAnything3Model: Module {
    let config: DA3Config
    let backbone: DinoV2Backbone
    let head: DualDPTHead
    let camEnc: CameraEncoder

    public init(config: DA3Config = .da3Small) {
        self.config = config
        backbone = DinoV2Backbone(config: config)
        head = DualDPTHead(
            dimIn: config.headDimIn, patchSize: config.patchSize,
            outputDim: config.headOutputDim, features: config.headFeatures,
            outChannels: config.headOutChannels)
        camEnc = CameraEncoder(dimOut: config.camEncDimOut)
    }

    /// Load model weights from a safetensors file (MLX format).
    public func loadWeights(from path: URL) throws {
        let weights = try MLX.loadArrays(url: path)
        try update(parameters: ModuleParameters.unflattened(weights))
        eval(parameters())
    }

    /// Run depth prediction.
    ///
    /// - Parameters:
    ///   - images: (B, S, H, W, 3) NHWC float32, ImageNet-normalised
    ///   - extrinsics: optional (B, S, 4, 4) world-to-camera matrices
    ///   - intrinsics: optional (B, S, 3, 3) camera intrinsics
    /// - Returns: (depth: (B, S, H, W), depthConf: (B, S, H, W))
    public func callAsFunction(
        images: MLXArray,
        extrinsics: MLXArray? = nil,
        intrinsics: MLXArray? = nil
    ) -> (depth: MLXArray, depthConf: MLXArray) {
        let (_, _, h, w, _) = (
            images.shape[0], images.shape[1], images.shape[2],
            images.shape[3], images.shape[4]
        )

        var camToken: MLXArray?
        if let ext = extrinsics, let ixt = intrinsics {
            camToken = camEnc(ext: ext.asType(.float32), ixt: ixt.asType(.float32),
                              imageSize: (h, w))
        }

        let feats = backbone(images, camToken: camToken)
        let result = head(feats: feats, h: h, w: w)
        return result
    }
}
