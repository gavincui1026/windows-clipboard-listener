# Profanity地址格式指南

## 地址格式（20字符匹配模式）

profanity使用20个字符的匹配模式来生成地址。

### 基本格式
```
T + 中间任意字符（用X表示）+ 想要匹配的字符 = 20个字符
```

### 示例

#### 1. 后缀匹配（20位格式 - 推荐）
- 后5位11111：`TXXXXXXXXXXXXXXX11111`（T + 15个X + 5个1）
- 后6位BBBBBB：`TXXXXXXXXXXXXXBBBBBB`（T + 14个X + 6个B）
- 后4位1234：`TXXXXXXXXXXXXXXXX1234`（T + 16个X + 4个数字）

#### 2. 后缀匹配（34位格式）
- 后5位11111：`TXXXXXXXXXXXXXXXXXXXXXXXXXXXXX11111`（T + 29个X + 5个1）

#### 3. 前缀+后缀匹配
- 前2位AB，后5位AAAAA：`TABXXXXXXXXXXXXAAAAA`（T + AB + 12个X + 5个A）

### 命令示例
```bash
# 生成后5位为11111的地址（20位格式）
profanity --matching TXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1

# 生成后6位为AAAAAA的地址（20位格式）
profanity --matching TXXXXXXXXXXXXXAAAAAA --suffix-count 6 --quit-count 1

# 生成后5位为11111的地址（34位格式）
profanity --matching TXXXXXXXXXXXXXXXXXXXXXXXXXXXXX11111 --suffix-count 5 --quit-count 1
```

### 注意事项
1. 匹配模式可以是20个字符（推荐）或34个字符
2. 第一个字符必须是T（TRON地址前缀）
3. X表示任意字符
4. `--suffix-count`参数应该与后缀长度匹配
5. 推荐使用20位格式，更简洁！

## 代码实现
```python
suffix_pattern = address[-5:]  # 获取后5位
# 对于20位格式：后5位需要15个X
x_count = 20 - len(suffix_pattern)  # 20 - 5 = 15
matching_pattern = "T" + "X" * x_count + suffix_pattern
# 结果：TXXXXXXXXXXXXXXX11111
```
