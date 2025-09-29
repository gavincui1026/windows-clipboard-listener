-- 为防止循环生成添加必要的数据库索引

-- 为generated_address字段添加索引，用于快速检查地址是否已生成
CREATE INDEX IF NOT EXISTS idx_generated_address 
ON generated_addresses(generated_address);

-- 查看所有索引
-- SHOW INDEXES FROM generated_addresses;
