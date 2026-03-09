// swift-tools-version: 5.9
// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

import PackageDescription

let package = Package(
    name: "DepthAnything3MLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "DepthAnything3MLX",
            targets: ["DepthAnything3MLX"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "DepthAnything3MLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/DepthAnything3MLX"
        ),
    ]
)
