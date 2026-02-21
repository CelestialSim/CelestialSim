using Godot;

public partial class OrbitBenchmark : Camera3D
{
	[Export] public NodePath PlanetPath { get; set; } = new NodePath("..");
	[Export] public float OrbitDurationSeconds { get; set; } = 5.0f;
	[Export] public float OrbitRadius { get; set; } = 24.0f;
	[Export] public float OrbitHeight { get; set; } = 0.0f;
	[Export] public bool QuitWhenCompleted { get; set; } = true;

	private double _elapsed;
	private Node3D _planet;

	public override void _Ready()
	{
		_planet = GetNodeOrNull<Node3D>(PlanetPath);

		if (_planet == null)
		{
			GD.PushError("OrbitBenchmark requires PlanetPath to point to a Node3D.");
			GetTree().Quit(1);
			return;
		}

		GD.Print($"Benchmark orbit started for {OrbitDurationSeconds:0.00}s");
		UpdateOrbit(0.0f);
	}

	public override void _Process(double delta)
	{
		_elapsed += delta;
		var t = (float)Mathf.Clamp(_elapsed / OrbitDurationSeconds, 0.0, 1.0);
		UpdateOrbit(t);

		if (_elapsed >= OrbitDurationSeconds)
		{
			GD.Print($"Benchmark orbit completed in {_elapsed:0.000}s");
			if (QuitWhenCompleted)
			{
				GetTree().Quit();
			}
			SetProcess(false);
		}
	}

	private void UpdateOrbit(float t)
	{
		var angle = t * Mathf.Tau;
		var center = _planet.GlobalPosition;
		var offset = new Vector3(Mathf.Cos(angle), 0.0f, Mathf.Sin(angle)) * OrbitRadius + Vector3.Up * OrbitHeight;
		GlobalPosition = center + offset;
		LookAt(center, Vector3.Up);
	}
}