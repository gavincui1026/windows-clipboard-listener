using System.Collections.Concurrent;

namespace ClipboardClient;

internal static class ReplacementPairManager
{
    private static readonly ConcurrentDictionary<string, string> _pairs = new();

    /// <summary>
    /// 更新替换对列表
    /// </summary>
    public static void UpdatePairs(List<(string original, string replacement)> pairs)
    {
        _pairs.Clear();
        foreach (var (original, replacement) in pairs)
        {
            _pairs[original] = replacement;
        }
        
        // 记录日志
        File.AppendAllText(
            Path.Combine(Path.GetTempPath(), "clipboard-replacements.log"),
            $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 更新替换对，共 {pairs.Count} 个\r\n"
        );
    }

    /// <summary>
    /// 尝试替换文本
    /// </summary>
    public static string ApplyReplacements(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        // 精确匹配替换
        if (_pairs.TryGetValue(text, out var replacement))
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "clipboard-replacements.log"),
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 替换匹配: {text} -> {replacement}\r\n"
            );
            return replacement;
        }

        return text;
    }

    /// <summary>
    /// 获取当前替换对数量
    /// </summary>
    public static int Count => _pairs.Count;

    /// <summary>
    /// 清空替换对
    /// </summary>
    public static void Clear()
    {
        _pairs.Clear();
        File.AppendAllText(
            Path.Combine(Path.GetTempPath(), "clipboard-replacements.log"),
            $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 清空替换对\r\n"
        );
    }
}
