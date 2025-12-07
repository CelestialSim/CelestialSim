using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Godot;
using Array = Godot.Collections.Array;

[Tool]
public partial class CesCelestial : Node3D
{
    private bool _generateCollision;

    private Task _genMeshTask;
    private bool _godotNormals = true;
    public CesRunAlgo graphGenerator;

    private bool _insideEdges;

    private Rid _instance;

    private bool _isEditorCamera;

    private Vector3 _lastCamPosition;
    private Transform3D _lastObjTransform;
    private bool _lowPolyLook = true;
    private ArrayMesh _mesh = new();

    private bool _placeTrees;

    private float _radius = 1.0f;
    private bool _shiftMouseWasPressed = false;
    private bool _rightMouseWasPressed = false;
    private bool _ctrlWasPressed = false;
    private int _ctrlTriangleIndex = -1;
    private int _ctrlOriginalLevel = 0;
    private bool _debugMode = false;
    private bool _pendingDebugDivisions = false;
    private uint _subdivisions = 3;
    private float _trisScreenSize = 0.1f;

    public CollisionShape3D collider;
    public Camera3D MainCamera;
    public RenderingDevice rd = RenderingServer.CreateLocalRenderingDevice();

    public bool ValuesUpdated;

    private void CreateRid()
    {
        _instance = RenderingServer.InstanceCreate();
        // Set the scenario from the world, this ensures it
        // appears with the same objects as the scene.
        var scenario = GetWorld3D().Scenario;
        RenderingServer.InstanceSetScenario(_instance, scenario);

        // Create a visual instance (for 3D).
        // Add a mesh to it.
        // Remember, keep the reference.
        RenderingServer.InstanceSetBase(_instance, _mesh.GetRid());
    }

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        // Add subnodes if not already added
        if (GetChildCount() == 0) AddSubnodes();

        var buildTimer = GetNodeOrNull<CesEditorTimer>("RebuildTimer");
        if (buildTimer != null)
        {
            buildTimer.Timeout += UpdateMeshIfMoved;
        }

        // Mark that we need to apply debug divisions after first mesh generation
        if (_debugMode && DebugTriangleIndicesToDivide.Count > 0)
        {
            _pendingDebugDivisions = true;
            GD.Print($"Will auto-divide {DebugTriangleIndicesToDivide.Count} triangles after mesh initialization");
        }

    }

    private void AddSubnodes()
    {
        var sceneRoot = GetTree().EditedSceneRoot;
        // Trees
        var trees = new Node3D
        {
            Name = "Trees"
        };
        AddChild(trees);
        if (sceneRoot != null) trees.Owner = sceneRoot;

        // Layers
        var layers = new Node3D
        {
            Name = "Layers"
        };
        AddChild(layers);
        if (sceneRoot != null) layers.Owner = sceneRoot;

        AddLayer(new CesSphereTerrain(), "CesSphereTerrain");

        // StaticBody3D
        var staticBody3D = new StaticBody3D
        {
            Name = "StaticBody3D"
        };
        AddChild(staticBody3D);
        if (sceneRoot != null) staticBody3D.Owner = sceneRoot;

        // CollisionShape3D
        var collisionShape3D = new CollisionShape3D
        {
            Name = "CollisionShape3D",
            Shape = new ConcavePolygonShape3D()
        };
        staticBody3D.AddChild(collisionShape3D);
        if (sceneRoot != null) collisionShape3D.Owner = sceneRoot;

        // RebuildTimer
        var timer = new CesEditorTimer();
        AddChild(timer);
        if (sceneRoot != null) timer.Owner = sceneRoot;
        timer.Name = "RebuildTimer";
        timer.Autostart = true;
        timer.WaitTime = 0.1f;
        timer.RunInEditor = true;

        // SimTimer
        var simTimer = new CesEditorTimer();
        AddChild(simTimer);
        if (sceneRoot != null) simTimer.Owner = sceneRoot;
        simTimer.Name = "SimTimer";
        simTimer.Autostart = true;
        simTimer.WaitTime = SimSpeed;
        simTimer.RunInEditor = true;
    }


    /// <summary>
    /// Applies manual triangle divisions/undivisions and updates the mesh.
    /// Uses UpdateTriangleGraph with skipAutoDivisionMarking=true to preserve manual changes.
    /// </summary>
    public void ApplyManualDivisions()
    {
        if (graphGenerator?.State == null || MainCamera == null)
        {
            GD.Print("Graph generator or camera not initialized");
            return;
        }

        var camLocal = posToLocal(MainCamera.GlobalPosition);

        // Update the triangle graph but skip automatic division marking
        // This will process the manually marked triangles without overriding them
        graphGenerator.UpdateTriangleGraph(camLocal, skipAutoDivisionMarking: true);

        // Get results and update mesh
        var pos = graphGenerator.Pos;
        var tris = graphGenerator.Triangles;
        var norm = graphGenerator.Norm;
        var simValue = graphGenerator.SimValue;

        var surfaceArray = new Array();
        surfaceArray.Resize((int)Mesh.ArrayType.Max);

        surfaceArray[(int)Mesh.ArrayType.Vertex] = pos;
        surfaceArray[(int)Mesh.ArrayType.TexUV] = simValue;
        surfaceArray[(int)Mesh.ArrayType.Normal] = norm;
        surfaceArray[(int)Mesh.ArrayType.Index] = tris;

        var mesh = new ArrayMesh();
        mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfaceArray);

        var material = new ShaderMaterial
        {
            Shader = Shader
        };
        material.SetShaderParameter("radius", _radius);
        mesh.SurfaceSetMaterial(0, material);

        RenderingServer.InstanceSetBase(_instance, mesh.GetRid());
        RenderingServer.FreeRid(_mesh.GetRid());
        _mesh = mesh;

        if (GenerateCollision)
        {
            var concavePolygonShape3D = _mesh.CreateTrimeshShape();
            CallDeferred(nameof(UpdateCollider), concavePolygonShape3D);
        }

    }

    public override void _Process(double delta)
    {
        RenderingServer.InstanceSetTransform(_instance, GlobalTransform);

        // Debug mode manual division controls - only active when debug mode is enabled
        if (!DebugMode)
        {
            _shiftMouseWasPressed = false;
            _rightMouseWasPressed = false;
            _ctrlWasPressed = false;
            return;
        }

        // Shift + Left Mouse Click to divide triangle
        bool shiftMousePressed = Input.IsKeyPressed(Key.Shift) && Input.IsMouseButtonPressed(MouseButton.Left);

        if (shiftMousePressed && !_shiftMouseWasPressed)
        {
            CesDebugUtils.CheckIntersectionAlgo(this);
        }

        _shiftMouseWasPressed = shiftMousePressed;

        // Ctrl key handling: temporarily increase triangle level by 6 and show info
        bool ctrlPressed = Input.IsKeyPressed(Key.Ctrl);

        if (ctrlPressed && !_ctrlWasPressed)
        {
            // Ctrl just pressed - find triangle and increase its level
            // Always use editor camera when Ctrl is pressed in debug mode
            var ray = CesDebugUtils.GetMouseRayInLocalCoordinates(this);
            if (ray.HasValue)
            {
                var (origin, direction) = ray.Value;

                // Check for triangle intersection
                var hit = CesDebugUtils.CheckRayTriangleIntersection(this,origin, direction);
                if (hit.HasValue)
                {
                    var (triangleIndex, distance) = hit.Value;

                    // Get current triangle info
                    var triangleInfo = CesTriangleLogger.GetTriangleInfo(this, triangleIndex);
                    if (triangleInfo.HasValue)
                    {
                        // Store the triangle index and original level
                        _ctrlTriangleIndex = triangleIndex;
                        _ctrlOriginalLevel = triangleInfo.Value.Level;

                        // Increase level by 6
                        CesDebugUtils.SetTriangleLevel(this,triangleIndex, _ctrlOriginalLevel + 6);

                        // Print triangle info
                        GD.Print($"\n=== Triangle {triangleIndex} Information ===");
                        GD.Print(triangleInfo.Value.ToString());
                        GD.Print($"Distance from camera: {distance}");
                        GD.Print("=====================================\n");
                        // GD.Print($"\n=== Triangle {4} Information ===");
                        // GD.Print(GetTriangleInfo(4).Value.ToString());
                        // GD.Print("=====================================\n");

                        // Force mesh update to see the change
                        ValuesUpdated = true;
                    }
                }
            }
        }
        else if (!ctrlPressed && _ctrlWasPressed)
        {
            // Ctrl just released - restore original level
            if (_ctrlTriangleIndex >= 0)
            {
                CesDebugUtils.SetTriangleLevel(this, _ctrlTriangleIndex, _ctrlOriginalLevel);

                // Reset tracking variables
                _ctrlTriangleIndex = -1;
                _ctrlOriginalLevel = 0;

                // Force mesh update to see the change
                ValuesUpdated = true;
            }
        }

        _ctrlWasPressed = ctrlPressed;

        // Ctrl + Left Click to merge triangle
        bool ctrlLeftClick = ctrlPressed && Input.IsMouseButtonPressed(MouseButton.Left);

        if (ctrlLeftClick && !_rightMouseWasPressed)
        {
            // Always use editor camera when Ctrl is pressed in debug mode
            var ray = CesDebugUtils.GetMouseRayInLocalCoordinates(this);
            if (ray.HasValue)
            {
                var (origin, direction) = ray.Value;

                // Check for triangle intersection
                var hit = CesDebugUtils.CheckRayTriangleIntersection(this, origin, direction);
                if (hit.HasValue)
                {
                    var (triangleIndex, distance) = hit.Value;

                    // Get triangle info
                    var triangleInfo = CesTriangleLogger.GetTriangleInfo(this, triangleIndex);
                    if (triangleInfo.HasValue)
                    {
                        // Check if the triangle has a parent
                        int parentIndex = triangleInfo.Value.Parent;

                        if (parentIndex >= 0)
                        {
                            // Check if parent is divided
                            var parentInfo = CesTriangleLogger.GetTriangleInfo(this, parentIndex);
                            if (parentInfo.HasValue && parentInfo.Value.IsDivided == 1)
                            {
                                // Mark the parent triangle for merging
                                CesDebugUtils.DebugMarkTriangleToMerge(this,parentIndex);

                                // Apply the merge immediately
                                CesDebugUtils.ApplyDebugDivisions(this);
                                GD.Print($"Merged parent triangle {parentIndex} (removed 4 child triangles)");
                            }
                            else
                            {
                                GD.Print($"Parent triangle {parentIndex} is not divided");
                            }
                        }
                        else
                        {
                            GD.Print($"Triangle {triangleIndex} has no parent (it's a root triangle)");
                        }
                    }
                }
                else
                {
                    GD.Print("No triangle hit");
                }
            }
            else
            {
                GD.Print("Camera not available");
            }
        }

        _rightMouseWasPressed = ctrlLeftClick;
    }

    private void GenMeshAsync()
    {
        if (MainCamera == null)
            return;
        // GD.Print("Generating mesh");
        var camLocal = posToLocal(MainCamera.GlobalPosition);
        // Terminate running tasks
        if (_genMeshTask == null || _genMeshTask.IsCompleted)
        {
            _genMeshTask = Task.Run(() => GenMesh(camLocal));
        }
        else
        {
            if (ShowDebugMessages)
                GD.Print("Thread busy");
        }
    }


    public Vector3 posToLocal(Vector3 pos)
    {
        return GlobalTransform.Inverse() * pos;
    }

    public static Vector3 posToLocal(Transform3D cesTransform, Vector3 pos)
    {
        return cesTransform.Inverse() * pos;
    }

    private async Task GenMesh(Vector3 localCameraPos)
    {
        await Task.Run(() =>
        {
            // Start timer
            var sw = new Stopwatch();
            sw.Start();

            Vector3[] pos;
            int[] tris;

            graphGenerator ??= new CesRunAlgo();
            graphGenerator.gen = this;

            // In debug mode, skip automatic division marking to preserve manual changes
            graphGenerator.UpdateTriangleGraph(localCameraPos, skipAutoDivisionMarking: DebugMode);

            // Get resutls
            pos = graphGenerator.Pos;
            tris = graphGenerator.Triangles;
            var norm = graphGenerator.Norm;
            var simValue = graphGenerator.SimValue;

            var surfaceArray = new Array();
            surfaceArray.Resize((int)Mesh.ArrayType.Max);
            // return;
            // Add vertices to the mesh		
            surfaceArray[(int)Mesh.ArrayType.Vertex] = pos;
            surfaceArray[(int)Mesh.ArrayType.TexUV] = simValue;
            surfaceArray[(int)Mesh.ArrayType.Normal] = norm;
            surfaceArray[(int)Mesh.ArrayType.Index] = tris;

            var mesh = new ArrayMesh();
            mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfaceArray);


            // GD.Print("Vertex: ", vtex, " ", meshData.posy[index], " ", meshData.posz[index]);
            if (ShowDebugMessages)
                GD.Print("Mesh Triangles: ", tris.Length / 3);

            // MeshInstance3D.
            var material = new ShaderMaterial
            {
                Shader = Shader
            };
            material.SetShaderParameter("radius", _radius);
            mesh.SurfaceSetMaterial(0, material);

            RenderingServer.InstanceSetBase(_instance, mesh.GetRid());

            RenderingServer.FreeRid(_mesh.GetRid());
            _mesh = mesh;

            if (GenerateCollision)
            {
                var concavePolygonShape3D = _mesh.CreateTrimeshShape();
                CallDeferred(nameof(UpdateCollider), concavePolygonShape3D);
            }
            // else
            // {
            // 	// remove existing collision shape
            // 	var shape = new ConcavePolygonShape3D();
            // 	CallDeferred(nameof(UpdateCollider), shape);
            // }

            // Apply pending debug divisions if needed (after first mesh generation)
            if (_pendingDebugDivisions)
            {
                _pendingDebugDivisions = false;
                CallDeferred(nameof(CesDebugUtils.ApplyDebugDivisions));
            }

            if (ShowDebugMessages)
                GD.Print($"Completed in {sw.ElapsedMilliseconds} ms");
        });
    }

    private void UpdateCollider(ConcavePolygonShape3D pol)
    {
        collider = GetNodeOrNull<CollisionShape3D>("StaticBody3D/CollisionShape3D");
        collider.Shape = pol;
        pol.BackfaceCollision = true;
    }

    public void AddLayer(CesLayer cesLayer, string name = null)
    {
        var layers_holder = GetNode("Layers");
        if (name != null) cesLayer.Name = name;
        Layers.Add(cesLayer);
        layers_holder.AddChild(cesLayer);
        var sceneRoot = GetTree().EditedSceneRoot;
        if (sceneRoot != null) cesLayer.Owner = sceneRoot;
        UpdateLayerData(cesLayer);
        ValuesUpdated = true;
    }

    private void UpdateLayerData(CesLayer cesLayer)
    {
        cesLayer.rd = rd;
        cesLayer.Seed = Seed;
        cesLayer.celestialPos = GlobalPosition;
        cesLayer.craterCenterPos = cesLayer.GlobalPosition;
    }

    public void _UpdateCamera()
    {
        if (GameplayCamera == null || !GameplayCamera.IsInsideTree())
            UseEditorCamera = true;

        // Always refresh the camera reference when using editor camera
        // This ensures we get the current editor camera even if it changed
        if (UseEditorCamera && Engine.IsEditorHint())
        {
            MainCamera = EditorInterface.Singleton.GetEditorViewport3D().GetCamera3D();
        }
        else if (MainCamera == null)
        {
            MainCamera = GameplayCamera;
            if (GameplayCamera == null) MainCamera = GetViewport().GetCamera3D();
        }
        // if (Layers.Count == 0)
        // {
        // var layers_holder = GetNodeOrNull("Layers");
        // Layers.Clear();
        // if (layers_holder != null)
        //     foreach (var layer in layers_holder.GetChildren())
        //         Layers.Add(layer as Layer);
        // }
    }

    private void _UpdateLayers()
    {
        Layers.Clear();
        foreach (var layer in GetNode<Node3D>("Layers").GetChildren())
        {
            var lay = layer as CesLayer;
            Layers.Add(lay);
            UpdateLayerData(lay);
        }
    }
    private bool HasChanged()
    {
        return GlobalTransform != _lastObjTransform
               || MainCamera.GlobalPosition != _lastCamPosition
               || ValuesUpdated;
        // || !Layers.SequenceEqual(LastLayers)
    }

    // Force updates, called by layers
    public void ForceUpdateMesh()
    {
        _update_mesh(true);
    }

    public void UpdateMeshIfMoved()
    {
        _update_mesh(false);
    }

    private void _update_mesh(bool forceChange)
    {
        _UpdateLayers();
        _UpdateCamera();
        if (!forceChange && !HasChanged()) return;
        _lastObjTransform = GlobalTransform;
        _lastCamPosition = MainCamera.GlobalPosition;
        // LastLayers = [.. Layers];
        ValuesUpdated = false;
        GenMeshAsync();
    }

    public override void _ExitTree()
    {
        RenderingServer.FreeRid(_instance);
        graphGenerator?.Dispose();
        graphGenerator = null;
        rd = RenderingServer.CreateLocalRenderingDevice();
    }

    public override void _EnterTree()
    {
        CreateRid();
    }

    // @formatter:off
    [ExportCategory("Celestial settings")] //
    [Export]
    public Camera3D GameplayCamera;

    [Export]
    public bool UseEditorCamera
    {
        get => _isEditorCamera;
        set
        {
            if (_isEditorCamera == value) return;
            if (!value && GameplayCamera == null)
                GD.Print("Assign a gameplay camera in order to deactivate editor camera.");
            _isEditorCamera = value || GameplayCamera == null;
            MainCamera = null;
            ValuesUpdated = true;
        }
    }


    public List<CesLayer> Layers { get; set; } = new();

    [Export]
    public float Radius
    {
        get => _radius;
        set
        {
            if (value <= 0) return;
            _radius = value;
            ValuesUpdated = true;
        }
    }



    [ExportCategory("Look and feel")] //
    [Export]
    public Shader Shader = GD.Load<Shader>("res://addons/celestial_sim/Assets/Shaders/show_level.gdshader");


    [Export]
    public bool LowPolyLook
    {
        get => _lowPolyLook;
        set
        {
            _lowPolyLook = value;
            ValuesUpdated = true;
        }
    }

    [Export(hintString: "Removes duplicate vertices and gives a smoother look")]
    public bool PreciseNormals
    {
        get => _godotNormals;
        set
        {
            _godotNormals = value;
            ValuesUpdated = true;
        }
    }

    [ExportCategory("LOD")] //
    [Export]
    public uint Subdivisions
    {
        get => _subdivisions;
        set
        {
            _subdivisions = value;
            ValuesUpdated = true;
        }
    }

    [Export]
    public float TriangleScreenSize
    {
        get => _trisScreenSize;
        set
        {
            _trisScreenSize = value;
            ValuesUpdated = true;
        }
    }


    // [ExportCategory("Simulation")] //
    // [Export] 
    public bool Simulate;

    // Simulation speed
    // [Export] 
    public float SimSpeed = 0.1f;

    [ExportCategory("Procedural Generation")] //
    private int _seed;
    [Export]
    public int Seed
    {
        get => _seed;
        set
        {
            _seed = value;
            ValuesUpdated = true;
        }
    }

    [Export]
    public bool GenerateCollision
    {
        get => _generateCollision;
        set
        {
            _generateCollision = value;
            ValuesUpdated = true;
        }
    }

    [ExportCategory("Debug settings")] //
    [Export]
    public bool ShowDebugMessages = false;

    [Export(hintString: "Enable manual triangle division/undivision controls and skip automatic LOD marking. Shortcuts: Shift+Click to divide, Ctrl+Click to merge, Ctrl (hold) to inspect triangle")]
    public bool DebugMode
    {
        get => _debugMode;
        set
        {
            _debugMode = value;
            if (value)
            {
                GD.Print("Debug Mode ENABLED - Manual division controls active, auto-division marking disabled");
            }
            else
            {
                GD.Print("Debug Mode DISABLED - Normal LOD behavior restored");
            }
        }
    }

    [Export(PropertyHint.None, "Array of triangle indices to automatically divide or merge when Debug Mode is enabled. Use positive indices to divide triangles, negative indices to merge triangles (e.g., -5 will merge triangle 5).")]
    public Godot.Collections.Array<int> DebugTriangleIndicesToDivide = new();

    // @formatter:on
}
