using Godot;

[Tool]
public partial class CesEditorTimer : Timer
{
    private bool _runInEditor;

    [Export]
    public bool RunInEditor
    {
        get => _runInEditor;
        set
        {
            if (value != _runInEditor)
            {
                _runInEditor = value;
                if (Engine.IsEditorHint() && IsInsideTree())
                {
                    if (_runInEditor)
                        Start();
                    else
                        Stop();
                }
            }
        }
    }

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        // Replace with function body.
    }

    public override void _EnterTree()
    {
        if (Engine.IsEditorHint() && _runInEditor)
            Start();
    }
}