pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // 捆绑 SQLite（io.requery:sqlite-android，内置 FTS5）托管在 jitpack
        maven(url = "https://jitpack.io")
    }
}

rootProject.name = "LocalMind"
include(":app")
