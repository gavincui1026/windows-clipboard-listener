# TRON地址后缀匹配功能更新

## 更新内容

### 1. 功能描述
- 修改了TRON地址生成逻辑，从前缀匹配改为后缀匹配
- 生成的地址后5位与原始地址相同
- 使用profanity-tron工具实现，支持GPU加速

### 2. 技术实现

#### 新增函数
在 `app/utils/vanitygen_plusplus.py` 中新增了 `generate_trx_with_profanity` 函数：
- 自动检测系统平台，选择正确的可执行文件
- 构建匹配模式：`TXXXXXXXXXXXXXXXXXXXXXXXXXXAAAAA`（X表示任意字符）
- 使用参数：`--matching` + `--suffix-count 5` + `--quit-count 1`

#### 修改逻辑
修改了 `generate_trx_with_vpp` 函数：
1. 优先使用profanity-tron进行后缀匹配
2. 如果失败，自动回退到vanitygen-plusplus的前缀匹配

### 3. 文件变更
- `app/utils/vanitygen_plusplus.py` - 核心逻辑修改
- `test_suffix_generation.py` - 测试脚本
- `后缀匹配地址生成说明.md` - 详细说明文档
- `README.md` - 更新了功能说明

### 4. 使用示例

命令行直接测试：
```bash
profanity --matching TXXXXXXXXXXXXXXAAAAA --suffix-count 5 --quit-count 1
```

Python测试：
```bash
cd vanity-service
python test_suffix_generation.py
```

### 5. API调用不变
系统会自动处理，API调用方式保持不变：
```python
result = await client.generate_sync(
    address="TT1LT2H34YMurdmW9Hkxuy2hCbxzekjF5N",
    timeout=30,
    use_gpu=True
)
```

### 6. 性能说明
- 后缀匹配比前缀匹配计算量更大
- 建议使用GPU加速以获得更好的性能
- 生成时间取决于后缀的复杂度

### 7. 注意事项
1. profanity已经注册为全局命令，可在任何目录直接使用
2. 需要NVIDIA GPU和CUDA环境支持以获得最佳性能
3. 输出格式：地址和私钥在同一行，空格分隔
4. 程序会在找到匹配地址后自动退出
