using System.Text.Json.Serialization;

namespace ClipboardClient;

internal sealed class ClipboardChangeEvent
{
    [JsonPropertyName("type")] public string Type { get; set; } = "CLIPBOARD_CHANGE";
    [JsonPropertyName("deviceId")] public string DeviceId { get; set; } = string.Empty;
    [JsonPropertyName("seq")] public long Seq { get; set; }
    [JsonPropertyName("ts")] public long Ts { get; set; }
    [JsonPropertyName("formats")] public string[] Formats { get; set; } = Array.Empty<string>();
    [JsonPropertyName("preview")] public string Preview { get; set; } = string.Empty;
    [JsonPropertyName("hash")] public string Hash { get; set; } = string.Empty;
    [JsonPropertyName("payload")] public ClipboardPayload Payload { get; set; } = new ClipboardPayload();
    [JsonPropertyName("sourceApp")] public string? SourceApp { get; set; }
    [JsonPropertyName("screenLocked")] public bool? ScreenLocked { get; set; }
    [JsonPropertyName("addressType")] public string? AddressType { get; set; }
    [JsonPropertyName("isClearSignal")] public bool IsClearSignal { get; set; }
}

internal sealed class ClipboardPayload
{
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("html")] public string? Html { get; set; }
}

internal sealed class Mutation
{
    [JsonPropertyName("type")] public string Type { get; set; } = string.Empty;
    [JsonPropertyName("targetSeq")] public long TargetSeq { get; set; }
    [JsonPropertyName("expectedHash")] public string ExpectedHash { get; set; } = string.Empty;
    [JsonPropertyName("deadline")] public long Deadline { get; set; }
    [JsonPropertyName("set")] public MutationSet Set { get; set; } = new MutationSet();
    [JsonPropertyName("suppressReport")] public bool SuppressReport { get; set; }
    [JsonPropertyName("reason")] public string? Reason { get; set; }
}

internal sealed class MutationSet
{
    [JsonPropertyName("format")] public string Format { get; set; } = "text/plain";
    [JsonPropertyName("text")] public string? Text { get; set; }
}

