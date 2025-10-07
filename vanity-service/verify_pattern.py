#!/usr/bin/env python3
"""
验证匹配模式的正确性
"""

def verify_pattern(suffix_len):
    """验证不同后缀长度的模式"""
    # 根据用户的说明：
    # 后5位：T + 15个X + 5位后缀 = 20字符
    x_count = 20 - suffix_len
    
    # 构建模式
    pattern = "T" + "X" * x_count + "1" * suffix_len
    
    print(f"\n后缀长度: {suffix_len}")
    print(f"X的数量: {x_count}")
    print(f"模式: {pattern}")
    print(f"模式长度: {len(pattern)}")
    print(f"分解: T(1) + X({x_count}) + 后缀({suffix_len}) = {1 + x_count + suffix_len}")
    
    # 验证一些例子
    if suffix_len == 5:
        expected = "TXXXXXXXXXXXXXXX11111"
        print(f"预期: {expected}")
        print(f"匹配: {'✓' if pattern == expected else '✗'}")
    
    return pattern


def main():
    print("=== Profanity匹配模式验证 ===")
    print("\n根据用户说明：")
    print("20位格式：T + 15个X + 5个1 = 20字符")
    print("34位格式：T + 29个X + 5个1 = 34字符")
    
    # 测试不同的后缀长度
    for suffix_len in [4, 5, 6, 7]:
        verify_pattern(suffix_len)
    
    # 验证用户的具体例子
    print("\n\n=== 用户示例验证 ===")
    user_example = "TXXXXXXXXXXXXXXX11111"
    print(f"用户示例: {user_example}")
    print(f"长度: {len(user_example)}")
    
    # 分解计数
    t_count = 1
    x_count = user_example.count('X')
    one_count = user_example.count('1')
    
    print(f"T: {t_count}个")
    print(f"X: {x_count}个")
    print(f"1: {one_count}个")
    print(f"总计: {t_count} + {x_count} + {one_count} = {t_count + x_count + one_count}")


if __name__ == "__main__":
    main()
