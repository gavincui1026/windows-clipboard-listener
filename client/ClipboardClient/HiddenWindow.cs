using System.Runtime.InteropServices;

namespace ClipboardClient;

public partial class HiddenWindow : Form
{
    private readonly ClipboardWatcher _watcher;
    private readonly ClipboardService _service;
    private readonly AppConfig _config;
    private readonly string _deviceId;

    public HiddenWindow()
    {
        // 设置窗口属性，确保完全隐藏
        ShowInTaskbar = false;
        WindowState = FormWindowState.Minimized;
        FormBorderStyle = FormBorderStyle.None;
        Visible = false;
        
        // 初始化配置和服务
        _config = AppConfig.Load();
        _deviceId = DeviceIdStore.GetOrCreate();
        _service = new ClipboardService(_config, _deviceId);
        WebSocketManager.Instance.Initialize(_config, _deviceId, this);
        _watcher = new ClipboardWatcher(this, _service);
    }
    
    // 重写此方法以确保窗口永远不可见
    protected override void SetVisibleCore(bool value)
    {
        base.SetVisibleCore(false);
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        try 
        { 
            _watcher?.Dispose(); 
            // WebSocketManager 会自动管理连接
        } 
        catch { }
        base.OnFormClosed(e);
    }
}

internal sealed class ClipboardWatcher
{
    private readonly HiddenWindow _window;
    private readonly IntPtr _hwnd;
    private readonly ClipboardService _service;

    public ClipboardWatcher(HiddenWindow window, ClipboardService service)
    {
        _window = window;
        _hwnd = window.Handle;
        _service = service;
        NativeMethods.AddClipboardFormatListener(_hwnd);
    }

    public void Dispose()
    {
        NativeMethods.RemoveClipboardFormatListener(_hwnd);
    }

    private static class NativeMethods
    {
        public const int WM_CLIPBOARDUPDATE = 0x031D;

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool AddClipboardFormatListener(IntPtr hwnd);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool RemoveClipboardFormatListener(IntPtr hwnd);

        [DllImport("user32.dll")]
        public static extern uint GetClipboardSequenceNumber();
    }

    public void OnMessage(ref Message m)
    {
        if (m.Msg == NativeMethods.WM_CLIPBOARDUPDATE)
        {
            _service.OnClipboardUpdate();
        }
    }
}

