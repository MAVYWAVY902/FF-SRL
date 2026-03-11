"""
快速测试优化效果
"""
import subprocess
import time

print("="*70)
print("🚀 测试优化效果")
print("="*70)
print("\n正在运行优化后的代码...")
print("按 Ctrl+C 停止测试\n")
print("观察 FPS 和 Step 时间：")
print("  - 优化前：Step ~1740ms, FPS ~0.6")
print("  - 优化后：Step ~1-5ms,  FPS ~5-10")
print("="*70 + "\n")

try:
    subprocess.run([
        "python", "testInteractiveControl.py"
    ], cwd="/home/yunxin/FF-SRL/FF_SRL/FF_SRL/tests")
except KeyboardInterrupt:
    print("\n\n测试结束")
