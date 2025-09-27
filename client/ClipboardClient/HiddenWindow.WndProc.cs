namespace ClipboardClient;

public partial class HiddenWindow
{
    protected override void WndProc(ref Message m)
    {
        if (_watcher != null)
        {
            _watcher.OnMessage(ref m);
        }
        base.WndProc(ref m);
    }
}

