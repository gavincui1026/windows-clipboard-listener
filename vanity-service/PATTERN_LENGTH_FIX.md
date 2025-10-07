# Profanity匹配模式长度修正

## 问题发现
用户指出匹配模式的长度差异：
- 原代码：`TXXXXXXXXXXXXXXXXXXXXXXXXXXAAAAA`（34字符）
- 正确格式：`TXXXXXXXXXXXXXAAAAA`（20字符）

## 原因分析
profanity的`--matching`参数不需要指定完整的34字符TRON地址，只需要指定部分匹配模式。

## 修正内容

### 1. 匹配模式构建
```python
# 修正前：T + 28个X + 5个后缀 = 34个字符
matching_pattern = "T" + "X" * 28 + suffix_pattern

# 修正后：T + 14个X + 5个后缀 = 20个字符
matching_pattern = "T" + "X" * 14 + suffix_pattern
```

### 2. 为什么是20个字符？
- TRON地址总长度：34个字符
- profanity只需要部分匹配
- 20个字符的模式足够指定前缀和后缀
- 中间的字符由profanity随机生成

## 示例
```bash
# 生成后5位为AAAAA的地址
profanity --matching TXXXXXXXXXXXXXAAAAA --suffix-count 5 --quit-count 1

# 输出示例
TKiXgUWiRMbLWXeXKN3QK5AXzoY4gAAAAA f6882cb6b8be0e469a82649c9618e654354a79cae6c4d3da80b68783fb63798b
```

## 总结
这个修正让profanity能更高效地生成地址，因为匹配模式更短，搜索空间更合理。
