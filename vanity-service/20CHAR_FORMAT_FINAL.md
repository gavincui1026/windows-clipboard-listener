# Profanity 20字符格式最终说明

## 正确的格式规则

感谢用户的详细说明，现在已经正确实现了profanity的20字符匹配模式。

### 格式要求
- **总长度**：必须正好是20个字符
- **格式**：`T + 中间任意字符（X） + 想要的后缀`

### 示例计算
| 后缀长度 | X的数量 | 示例模式 |
|---------|--------|----------|
| 4位 | 15 | `TXXXXXXXXXXXXXXX1234` |
| 5位 | 14 | `TXXXXXXXXXXXXXXAAAAA` |
| 6位 | 13 | `TXXXXXXXXXXXXXBBBBBB` |
| 7位 | 12 | `TXXXXXXXXXXXXCCCCCCC` |

### 代码实现
```python
suffix_pattern = address[-5:]  # 获取后5位
x_count = 20 - 1 - len(suffix_pattern)  # 20 - T - 后缀长度
matching_pattern = "T" + "X" * x_count + suffix_pattern
```

### 正确的命令
```bash
# ✓ 正确（20字符）
profanity --matching TXXXXXXXXXXXXXXAAAAA --suffix-count 5 --quit-count 1

# ✗ 错误（19字符）- 会报错
profanity --matching TXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1

# ✗ 错误（34字符）- 太长
profanity --matching TXXXXXXXXXXXXXXXXXXXXXXXXXXAAAAA --suffix-count 5 --quit-count 1
```

### 测试验证
运行 `test_20char_format.py` 来验证不同后缀长度的模式构建：
```bash
cd vanity-service
python test_20char_format.py
```

## 总结
现在系统能正确构建20字符的匹配模式，确保profanity能成功生成后缀匹配的TRON地址。
