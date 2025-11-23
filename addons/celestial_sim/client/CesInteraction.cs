using Godot;

namespace ces.Rendering;


// This class handles interaction between the player and the celestion
// Currently collision generation placing objects on the surface with a mouse click and 
// finding the closest triangle to a point methods are handled
public class CesInteraction
{
    
	private static Vector3? GetRayCastWithTerrain(CesCelestial celestial)
	{
		Camera3D editorCamera = celestial.MainCamera;
		var spaceState = celestial.GetWorld3D().DirectSpaceState;

		if (editorCamera.GetViewport() is SubViewport viewport &&
			viewport.GetParent() is SubViewportContainer viewportContainer)
		{
			var screenPosition = editorCamera.GetViewport().GetMousePosition();

			var from = editorCamera.ProjectRayOrigin(screenPosition);
			var dir = editorCamera.ProjectRayNormal(screenPosition);

			var distance = 2000;
			var query = new PhysicsRayQueryParameters3D
			{
				From = from,
				To = from + dir * distance
			};
			var result = spaceState.IntersectRay(query);

			GD.Print($"result: {result}");
			// DrawLine(from, from + dir * distance);
			if (result?.Count > 0 && result["collider"].Obj == celestial.collider.GetParent()) return (Vector3)result["position"];
		}

		return null;
	}

	public static Transform3D? GetPointOnPlanet(CesCelestial celestial)
	{
		// Returns a transform: The position the result of the raycast with the celestial.
		// The direction is the radial direction in the -Z axis
		var v3 = GetRayCastWithTerrain(celestial);
		if (v3.HasValue)
		{
			var pos = v3.Value;

			var normal = (pos - celestial.GlobalPosition).Normalized();

			Transform3D pointTransform = new Transform3D();
			var upAxis = Vector3.Up.Cross(normal).Normalized();
			pointTransform.Origin = pos;
			pointTransform = pointTransform.LookingAt(pos + normal, upAxis);
			return pointTransform;
			
			// tree.Basis = tree.Basis.Rotated(tree.Basis.X, -Mathf.Pi / 2);
		}
		else return null;
	}

}