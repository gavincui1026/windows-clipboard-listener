/*
 * 简单测试 - 验证地址生成和匹配逻辑
 */
#include <stdio.h>
#include <string.h>

// 简化的模式匹配测试
bool test_pattern_match(const char* address, const char* prefix, const char* suffix) {
    printf("测试匹配:\n");
    printf("  地址: %s\n", address);
    printf("  前缀: %s (跳过'T')\n", prefix);
    printf("  后缀: %s\n", suffix);
    
    // 从索引1开始匹配前缀（跳过'T'）
    int prefix_len = strlen(prefix);
    for (int i = 0; i < prefix_len; i++) {
        if (address[i + 1] != prefix[i]) {
            printf("  前缀不匹配: address[%d]='%c' != prefix[%d]='%c'\n", 
                   i+1, address[i+1], i, prefix[i]);
            return false;
        }
    }
    
    // 匹配后缀
    if (suffix[0] != '\0') {
        int addr_len = strlen(address);
        int suffix_len = strlen(suffix);
        
        for (int i = 0; i < suffix_len; i++) {
            if (address[addr_len - suffix_len + i] != suffix[i]) {
                printf("  后缀不匹配\n");
                return false;
            }
        }
    }
    
    printf("  ✓ 匹配成功!\n");
    return true;
}

int main() {
    // 测试案例
    const char* test_address = "T11xxxxxxxxxxxxxxxxxxxxxxxxxxxF5N";
    
    printf("=== 模式匹配测试 ===\n\n");
    
    // 测试1: 匹配 "11...F5N"
    test_pattern_match(test_address, "11", "F5N");
    
    printf("\n");
    
    // 测试2: 不匹配的情况
    test_pattern_match(test_address, "22", "F5N");
    
    printf("\n");
    
    // 测试3: 只匹配前缀
    test_pattern_match(test_address, "11", "");
    
    return 0;
}
