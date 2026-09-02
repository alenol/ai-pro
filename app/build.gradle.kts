plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// CI 通过 -Plocalmind.arm64Only=true 只编 arm64-v8a（见 .github/workflows/build-release.yml）
val arm64Only = (project.findProperty("localmind.arm64Only") as? String) == "true"

android {
    namespace = "com.localmind.ai"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.localmind.ai"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // 骁龙8 Gen5 用 NDK 27 系列，支持 C++17、KleidiAI
        ndkVersion = "27.0.12077973"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // CI 传 -Plocalmind.debugLog=true 时默认开启详细日志（日志写入 Download/localmodel/）
        val debugLog = (project.findProperty("localmind.debugLog") as? String) == "true"
        buildConfigField("boolean", "DEBUG_LOG", debugLog.toString())

        externalNativeBuild {
            cmake {
                // 让 llama.cpp 的 OpenCL 内核以 .cl 源码形式编进 so（见 cpp/CMakeLists.txt）
                arguments += "-DANDROID_STL=c++_shared"
            }
        }

        // 默认同时编 arm64-v8a（骁龙）与 x86_64（模拟器调试）。
        // CI 传 -Plocalmind.arm64Only=true 时只编 arm64-v8a，缩小产物体积、加快编译。
        ndk {
            abiFilters += if (arm64Only) listOf("arm64-v8a") else listOf("arm64-v8a", "x86_64")
        }
    }

    // CI / 本地签名：仅在提供了 LM_SIGN_STORE_FILE 环境变量时启用，
    // 否则不挂签名配置，避免本地无密钥时构建报错。
    signingConfigs {
        create("ci") {
            val storeFile = System.getenv("LM_SIGN_STORE_FILE")
            if (storeFile != null) {
                this.storeFile = file(storeFile)
                this.storePassword = System.getenv("LM_SIGN_STORE_PASSWORD")
                this.keyAlias = System.getenv("LM_SIGN_KEY_ALIAS")
                this.keyPassword = System.getenv("LM_SIGN_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            if (System.getenv("LM_SIGN_STORE_FILE") != null) {
                signingConfig = signingConfigs.getByName("ci")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        aidl = true
        // 生成 BuildConfig.DEBUG_LOG（用 -Plocalmind.debugLog=true 开启详细日志）
        buildConfig = true
    }

    composeOptions {
        // 使用 org.jetbrains.kotlin.plugin.compose 插件时无需手动指定 extension 版本
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")

    // Compose
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    // 测试
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}
