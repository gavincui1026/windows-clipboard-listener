# Profanity最终格式说明

## 正确的格式（根据用户确认）

### 20位格式（推荐）
对于后5位匹配：
```
profanity --matching TXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1
```
- T = 1个字符
- X = 15个
- 后缀 = 5个
- 总计 = 20个字符

### 34位格式
对于后5位匹配：
```
profanity --matching TXXXXXXXXXXXXXXXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1
```
- T = 1个字符
- X = 29个
- 后缀 = 5个
- 总计 = 34个字符

## X数量计算规则

对于20位格式：
- 后4位：X = 16个
- 后5位：X = 15个
- 后6位：X = 14个
- 后7位：X = 13个

计算公式：`x_count = 20 - suffix_length`

## 代码实现
```python
suffix_pattern = address[-5:]
x_count = 20 - len(suffix_pattern)  # 20 - 5 = 15
matching_pattern = "T" + "X" * x_count + suffix_pattern
```

## 验证例子

### 后5位为11111
```bash
# 20位格式（推荐）
profanity --matching TXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1

# 34位格式
profanity --matching TXXXXXXXXXXXXXXXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1
```

### 后6位为AAAAAA
```bash
# 20位格式
profanity --matching TXXXXXXXXXXXXXAAAAAA --suffix-count 6 --quit-count 1
```

## 总结
根据用户的明确指示，20位格式是推荐的格式，更简洁。代码已按照用户的正确格式实现。
