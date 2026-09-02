# LocalMind 混淆规则
# 调试期已关闭 minify，正式发布时再按需收紧。

# 保留 JNI 桥接类（方法名由 native 层按全限定名查找）
-keep class com.localmind.ai.engine.NativeEngine { *; }

# 保留 AIDL 生成类
-keep class com.localmind.ai.ILocalMindApi** { *; }
-keep class com.localmind.ai.ILocalMindStreamCallback** { *; }

# llama.cpp 通过反射/字符串查找符号，不要优化掉
-keep class com.localmind.ai.engine.** { *; }

# 不混淆 JSON 字段（虽然我们用 org.json 手工解析，这里留个保险）
-keepattributes Signature
-keepattributes *Annotation*
