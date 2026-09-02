package com.localmind.ai.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF8C7BFF),
    secondary = Color(0xFF4DD0C4),
    tertiary = Color(0xFFFFB74D),
    background = Color(0xFF0F1115),
    surface = Color(0xFF1A1D24),
    onPrimary = Color(0xFF0F1115),
    onBackground = Color(0xFFE6E8EC),
    onSurface = Color(0xFFE6E8EC),
)

private val LightColorScheme = lightColorScheme(
    primary = Color(0xFF5B3DF5),
    secondary = Color(0xFF009688),
    tertiary = Color(0xFFE65100),
    background = Color(0xFFF7F8FA),
    surface = Color(0xFFFFFFFF),
)

@Composable
fun LocalMindTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme,
        content = content,
    )
}
