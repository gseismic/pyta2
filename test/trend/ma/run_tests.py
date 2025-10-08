#!/usr/bin/env python3
"""
移动平均模块测试运行脚本
运行所有移动平均相关的测试
"""

import subprocess
import sys
import os

def run_tests():
    """运行所有移动平均测试"""
    test_files = [
        'test_sma.py',
        'test_ema.py', 
        'test_wma.py',
        'test_hma.py',
        'test_dema.py',
        'test_tema.py',
        'test_batch_functions.py',
        'test_api_functions.py'
    ]
    
    print("=" * 60)
    print("运行移动平均模块测试")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n运行 {test_file}...")
            print("-" * 40)
            
            try:
                result = subprocess.run(
                    ['pytest', test_file, '-v'],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                print(result.stdout)
                if result.stderr:
                    print("错误输出:")
                    print(result.stderr)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} 通过")
                    total_passed += 1
                else:
                    print(f"❌ {test_file} 失败")
                    total_failed += 1
                    
            except Exception as e:
                print(f"❌ 运行 {test_file} 时出错: {e}")
                total_failed += 1
        else:
            print(f"⚠️  文件 {test_file} 不存在")
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"总计: {total_passed + total_failed}")
    
    if total_failed == 0:
        print("🎉 所有测试都通过了！")
        return 0
    else:
        print("⚠️  有测试失败，请检查上述输出")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
