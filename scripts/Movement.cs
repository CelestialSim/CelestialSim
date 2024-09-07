using System;
using Godot;

public partial class Movement : CharacterBody3D
{
	[Export]
	public float Speed = 15.0f;
	[Export]
	public float JumpVelocity = 10f;
	[Export]
	public float MouseSensitivity = 0.2f;

	[Export]
	public float gravity = 0.0f;
	// ProjectSettings.GetSetting("physics/3d/default_gravity").AsSingle()
	[Export] private Camera3D Camera { get; set; }
	[Export] public Node3D Terrain { get; set; }
	[Export] public Label DebugLabel { get; set; }

	public override void _Ready()
	{
		base._Ready();
	}

	public override void _Input(InputEvent e)
	{
		if (e.IsPressed() && !e.IsEcho() && Input.IsKeyPressed(Key.Escape))
		{
			Input.MouseMode = Input.MouseMode == Input.MouseModeEnum.Captured ? Input.MouseModeEnum.Visible : Input.MouseModeEnum.Captured;
		}

		if (e is InputEventMouseMotion mouseMotion)
		{
			// Apply horizontal rotation around UpDirection
			float deltaRotationY = -mouseMotion.Relative.X * MouseSensitivity;
			Transform3D currentTransform = GlobalTransform;
			Basis rotationBasis = new Basis(UpDirection, Mathf.DegToRad(deltaRotationY));
			currentTransform.Basis = rotationBasis * currentTransform.Basis;
			GlobalTransform = currentTransform;

			// Apply vertical rotation to the camera
			Camera.RotateX(Mathf.DegToRad(-mouseMotion.Relative.Y * MouseSensitivity));

		}
	}

	private bool CanJump()
	{
		return true; //MotionMode == MotionModeEnum.Floating || IsOnFloor();
	}

	public override void _PhysicsProcess(double delta)
	{


		if (Terrain == null)
		{
			UpDirection = Vector3.Up;
		}
		else
		{
			UpDirection = (Position - Terrain.GlobalTransform.Origin).Normalized();
		}


		// GD.Print(GlobalTransform.Basis);


		Vector3 velocity = Transform.Basis.Inverse() * Velocity;
		// GD.Print(velLocal);

		if (!IsOnFloor())
		{
			velocity.Y -= gravity * (float)delta;
		}

		if (Input.IsActionJustPressed("ui_accept") && CanJump())
		{
			velocity.Y += JumpVelocity; //Math.Max(JumpVelocity, velocity.Y+JumpVelocity);
		}

		Vector2 inputDir = Input.GetVector("left", "right", "forward", "backward");
		Vector3 direction = (new Vector3(inputDir.X, 0, inputDir.Y)).Normalized();
		if (direction != Vector3.Zero)
		{
			velocity.X = direction.X * Speed;
			// velocity.Y = direction.Y * Speed + velocity.Y;
			velocity.Z = direction.Z * Speed;
		}
		else
		{
			velocity.X = Mathf.MoveToward(velocity.X, 0, Speed);
			velocity.Z = Mathf.MoveToward(velocity.Z, 0, Speed);
		}
		// Local velocity
		// GD.Print(velocity);

		Velocity = Transform.Basis * velocity;
		MoveAndSlide();

		var gt = GlobalTransform;
		var up = UpDirection;
		var right = -Basis.Z.Cross(up).Normalized();
		var forward = -up.Cross(right).Normalized();
		// gt.Basis.Y = UpDirection;
		// // newBasis.X = -UpDirection.Cross(Basis.Z).Normalized();
		// gt.Basis.Z = -gt.Basis.X.Cross(gt.Basis.Y).Normalized();
		// gt.Basis.X = -gt.Basis.Y.Cross(gt.Basis.Z).Normalized();
		gt = gt.LookingAt(gt.Origin + UpDirection, forward);
		gt.Basis = gt.Basis.Rotated(gt.Basis.X, -Mathf.Pi / 2);
		// Basis = newBasis;
		GlobalBasis = gt.Basis;
	}
}
