using System.Text.Json;

namespace ClipboardClient;

internal sealed class AppConfig
{
    public string WsUrl { get; set; } = "ws://156.251.17.161:8001/ws/clipboard";
    public string Jwt { get; set; } = "dev-token";
    public int AwaitMutationTimeoutMs { get; set; } = 500;
    public int SuppressMs { get; set; } = 800;
    public int ReconnectBaseDelayMs { get; set; } = 500;
    public int ReconnectMaxDelayMs { get; set; } = 5000;

    public static AppConfig Load()
    {
        var cfg = new AppConfig();

        var wsUrl = Environment.GetEnvironmentVariable("WS_URL");
        if (!string.IsNullOrWhiteSpace(wsUrl)) cfg.WsUrl = wsUrl!;

        var jwt = Environment.GetEnvironmentVariable("JWT");
        if (!string.IsNullOrWhiteSpace(jwt)) cfg.Jwt = jwt!;

        // 优先读取 config.json
        var configPath = Path.Combine(AppContext.BaseDirectory, "config.json");
        if (File.Exists(configPath))
        {
            try
            {
                var text = File.ReadAllText(configPath);
                var fileCfg = JsonSerializer.Deserialize<AppConfig>(text);
                if (fileCfg != null)
                {
                    MergeInto(cfg, fileCfg);
                }
            }
            catch
            {
                // ignore invalid config
            }
        }
        else
        {
            // 兼容旧版本，尝试读取 appsettings.json
            var appSettingsPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
            if (File.Exists(appSettingsPath))
            {
                try
                {
                    var text = File.ReadAllText(appSettingsPath);
                    var fileCfg = JsonSerializer.Deserialize<AppConfig>(text);
                    if (fileCfg != null)
                    {
                        MergeInto(cfg, fileCfg);
                    }
                }
                catch
                {
                    // ignore invalid config
                }
            }
        }

        return cfg;
    }

    private static void MergeInto(AppConfig target, AppConfig source)
    {
        if (!string.IsNullOrWhiteSpace(source.WsUrl)) target.WsUrl = source.WsUrl;
        if (!string.IsNullOrWhiteSpace(source.Jwt)) target.Jwt = source.Jwt;
        if (source.AwaitMutationTimeoutMs > 0) target.AwaitMutationTimeoutMs = source.AwaitMutationTimeoutMs;
        if (source.SuppressMs > 0) target.SuppressMs = source.SuppressMs;
        if (source.ReconnectBaseDelayMs > 0) target.ReconnectBaseDelayMs = source.ReconnectBaseDelayMs;
        if (source.ReconnectMaxDelayMs > 0) target.ReconnectMaxDelayMs = source.ReconnectMaxDelayMs;
    }
}

