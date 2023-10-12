// swift-tools-version: 5.8

// WARNING:
// This file is automatically generated.
// Do not edit it by hand because the contents will be replaced.

import PackageDescription
import AppleProductTypes

let package = Package(
    name: "AIplay",
    platforms: [
        .iOS("16.0")
    ],
    products: [
        .iOSApplication(
            name: "AIplay",
            targets: ["AppModule"],
            bundleIdentifier: "com.otabuzzman.aiplay.ios",
            teamIdentifier: "28FV44657B",
            displayVersion: "1.0.1",
            bundleVersion: "2",
            appIcon: .asset("AppIcon"),
            accentColor: .presetColor(.yellow),
            supportedDeviceFamilies: [
                .pad,
                .phone
            ],
            supportedInterfaceOrientations: [
                .portrait,
                .landscapeRight,
                .landscapeLeft,
                .portraitUpsideDown(.when(deviceFamilies: [.pad]))
            ],
            additionalInfoPlistContentFilePath: "Resources/AdditionalInfo.plist"
        )
    ],
    dependencies: [
        .package(url: "https://github.com/Losiowaty/PlaygroundTester.git", "0.3.1"..<"1.0.0"),
        .package(url: "https://github.com/mw99/DataCompression.git", "3.8.0"..<"4.0.0")
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            dependencies: [
                .product(name: "PlaygroundTester", package: "PlaygroundTester"),
                .product(name: "DataCompression", package: "DataCompression")
            ],
            path: ".",
            resources: [
                .process("Resources")
            ],
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .define("TESTING_ENABLED", .when(configuration: .debug))
            ]
        )
    ]
)
