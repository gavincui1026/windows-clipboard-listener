using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace ClipboardClient;

internal sealed class WebSocketManager
{
    private static readonly Lazy<WebSocketManager> _lazy = new(() => new WebSocketManager());
    public static WebSocketManager Instance => _lazy.Value;

    private readonly ConcurrentDictionary<long, TaskCompletionSource<Mutation?>> _awaiting = new();
    private readonly object _connectLock = new();
    private ClientWebSocket? _ws;
    private CancellationTokenSource? _cts;
    private Task? _recvLoop;
    private AppConfig _config = new();
    private string _deviceId = string.Empty;
    private Uri? _uri;
    private readonly JsonSerializerOptions _jsonOpts = new(JsonSerializerDefaults.Web);
    private readonly SemaphoreSlim _sendLock = new(1, 1);

    private Control? _mainControl;

    private WebSocketManager() { }

    public void Initialize(AppConfig config, string deviceId, Control mainControl)
    {
        _config = config;
        _deviceId = deviceId;
        _mainControl = mainControl;
        var baseUri = new Uri(config.WsUrl);
        var hasQuery = !string.IsNullOrEmpty(baseUri.Query);
        var sep = hasQuery ? "&" : "?";
        var escapedToken = Uri.EscapeDataString(config.Jwt);
        var escapedId = Uri.EscapeDataString(deviceId);
        var full = new Uri(baseUri + $"{sep}token={escapedToken}&deviceId={escapedId}");
        _uri = full;
        _ = EnsureConnectedAsync();
    }

    public async Task<Mutation?> SendEventAndAwaitAsync(ClipboardChangeEvent evt, int awaitTimeoutMs, CancellationToken ct)
    {
        await EnsureConnectedAsync();
        var payload = JsonSerializer.SerializeToUtf8Bytes(evt, _jsonOpts);
        var tcs = new TaskCompletionSource<Mutation?>(TaskCreationOptions.RunContinuationsAsynchronously);
        _awaiting[evt.Seq] = tcs;
        try
        {
            await _sendLock.WaitAsync(ct);
            try
            {
                if (_ws == null || _ws.State != WebSocketState.Open) return null;
                await _ws.SendAsync(new ArraySegment<byte>(payload), WebSocketMessageType.Text, true, ct);
            }
            finally
            {
                _sendLock.Release();
            }

            using var timeoutCts = new CancellationTokenSource(awaitTimeoutMs);
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);
            try
            {
                return await tcs.Task.WaitAsync(linked.Token);
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }
        finally
        {
            _awaiting.TryRemove(evt.Seq, out _);
        }
    }

    private async Task EnsureConnectedAsync()
    {
        if (_uri == null) return;
        if (_ws != null && _ws.State == WebSocketState.Open) return;
        lock (_connectLock)
        {
            if (_ws != null && _ws.State == WebSocketState.Open) return;
            _cts?.Cancel();
            _cts = new CancellationTokenSource();
            _ws = new ClientWebSocket();
        }

        var delay = _config.ReconnectBaseDelayMs;
        while (!_cts!.IsCancellationRequested)
        {
            try
            {
                await _ws!.ConnectAsync(_uri!, _cts.Token);
                
                // 记录连接成功
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] WebSocket连接成功: {_uri}\r\n"
                );
                
                _recvLoop = Task.Run(() => ReceiveLoopAsync(_cts.Token));
                _ = Task.Run(() => HeartbeatLoopAsync(_cts.Token));
                return;
            }
            catch (Exception ex)
            {
                // 记录连接失败
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] WebSocket连接失败: {ex.Message}\r\n"
                );
                
                await Task.Delay(delay, _cts.Token);
                delay = Math.Min(delay * 2, _config.ReconnectMaxDelayMs);
            }
        }
    }

    private async Task HeartbeatLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(20), ct);
                if (_ws == null || _ws.State != WebSocketState.Open) continue;
                var ping = Encoding.UTF8.GetBytes("{\"type\":\"PING\"}");
                await _sendLock.WaitAsync(ct);
                try
                {
                    await _ws.SendAsync(ping, WebSocketMessageType.Text, true, ct);
                }
                finally
                {
                    _sendLock.Release();
                }
            }
            catch
            {
                // ignore
            }
        }
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        var buffer = new byte[64 * 1024];
        var sb = new StringBuilder();
        while (!ct.IsCancellationRequested)
        {
            try
            {
                if (_ws == null) break;
                sb.Clear();
                WebSocketReceiveResult? result;
                do
                {
                    result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), ct);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", ct);
                        throw new WebSocketException("closed");
                    }
                    sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                } while (!result.EndOfMessage);

                var json = sb.ToString();
                HandleServerMessage(json);
            }
            catch
            {
                // force reconnect
                try { _ws?.Abort(); } catch { }
                await EnsureConnectedAsync();
            }
        }
    }

    private void HandleServerMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var type = root.GetProperty("type").GetString();
            if (string.Equals(type, "MUTATION", StringComparison.OrdinalIgnoreCase))
            {
                var mut = JsonSerializer.Deserialize<Mutation>(json, _jsonOpts);
                if (mut != null)
                {
                    if (_awaiting.TryGetValue(mut.TargetSeq, out var tcs))
                    {
                        tcs.TrySetResult(mut);
                    }
                }
            }
            else if (string.Equals(type, "PUSH_SET", StringComparison.OrdinalIgnoreCase))
            {
                // 记录收到推送消息
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 收到PUSH_SET消息: {json}\r\n"
                );
                
                // admin push set: apply immediately via ClipboardService (text only for now)
                var setElement = root.GetProperty("set");
                var format = setElement.TryGetProperty("format", out var fmtEl) ? fmtEl.GetString() : "text/plain";
                var text = setElement.TryGetProperty("text", out var txtEl) ? txtEl.GetString() : null;
                
                File.AppendAllText(
                    Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 解析内容 - format: {format}, text: {text}\r\n"
                );
                
                if (string.Equals(format, "text/plain", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(text))
                {
                    // 在UI线程上执行剪贴板操作
                    if (_mainControl?.InvokeRequired == true)
                    {
                        _mainControl.Invoke(() => ClipboardPush.ApplyExternalSet(text!));
                    }
                    else
                    {
                        ClipboardPush.ApplyExternalSet(text!);
                    }
                }
            }
            else if (string.Equals(type, "REPLACEMENT_PAIRS", StringComparison.OrdinalIgnoreCase))
            {
                // 处理替换对更新
                try
                {
                    var pairsArray = root.GetProperty("pairs");
                    var pairs = new List<(string original, string replacement)>();
                    
                    foreach (var pair in pairsArray.EnumerateArray())
                    {
                        var original = pair.GetProperty("original").GetString();
                        var replacement = pair.GetProperty("replacement").GetString();
                        
                        if (!string.IsNullOrEmpty(original) && replacement != null)
                        {
                            pairs.Add((original, replacement));
                        }
                    }
                    
                    ReplacementPairManager.UpdatePairs(pairs);
                    
                    File.AppendAllText(
                        Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                        $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 收到替换对更新，共 {pairs.Count} 个\r\n"
                    );
                }
                catch (Exception ex)
                {
                    File.AppendAllText(
                        Path.Combine(Path.GetTempPath(), "clipboard-push.log"),
                        $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 处理替换对失败: {ex.Message}\r\n"
                    );
                }
            }
        }
        catch
        {
            // ignore malformed
        }
    }
}


