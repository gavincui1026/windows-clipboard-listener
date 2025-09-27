namespace ClipboardClient;

internal static class DeviceIdStore
{
    public static string GetOrCreate()
    {
        var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        var dir = Path.Combine(appData, "WindowsClipboardListener");
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "device.id");
        if (File.Exists(file))
        {
            var id = File.ReadAllText(file).Trim();
            if (!string.IsNullOrWhiteSpace(id)) return id;
        }
        var newId = Guid.NewGuid().ToString("D");
        File.WriteAllText(file, newId);
        return newId;
    }
}

