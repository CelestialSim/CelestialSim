using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ces.Rendering.division;
using ces.Rendering.Sims;
using Godot;
using CelestialSim.scripts.client.Layers;
using Array = Godot.Collections.Array;

namespace ces.Rendering;

[Tool]
public partial class CesCelestial : Node3D
{
    private bool _generateCollision;

    private Task _genMeshTask;
    private bool _godotNormals = true;
    private CesClient _graphGenerator;

    private bool _insideEdges;

    private Rid _instance;
    private CesInteraction _interaction;

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
    private Task _simTask;
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

        var simTimer = GetNodeOrNull<CesEditorTimer>("SimTimer");
        if (simTimer != null)
        {
            simTimer.Timeout += SimulateAsync;
        }

        // Mark that we need to apply debug divisions after first mesh generation
        if (_debugMode && DebugTriangleIndicesToDivide.Count > 0)
        {
            _pendingDebugDivisions = true;
            GD.Print($"Will auto-divide {DebugTriangleIndicesToDivide.Count} triangles after mesh initialization");
        }

    }

    private async void Benchmark()
    {
        _subdivisions = 7;
        _trisScreenSize = 0f;
        GD.Print("Starting benchmark: ");
        var sw = new Stopwatch();
        sw.Start();
        const int repeat = 10;
        for (var i = 0; i < repeat; i++)
        {
            // GenMeshAsync();
            GD.Print($"Mesh {i + 1}/{repeat} generated");
            await Task.Delay(1000);
            PreciseNormals = !PreciseNormals;
        }

        sw.Stop();
        GD.Print("Time: ", sw.ElapsedMilliseconds);
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

        AddLayer(new CesNormalizePos(), "NormalizePos");

        // Simulation Layers
        var simLayers = new Node3D
        {
            Name = "SimLayers"
        };
        AddChild(simLayers);
        if (sceneRoot != null) simLayers.Owner = sceneRoot;

        AddSimLayer(new ShowLevelSimLayer(), "ShowLevelSim");

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

    public override void _Process(double delta)
    {
        RenderingServer.InstanceSetTransform(_instance, GlobalTransform);

        if (PlaceTrees && Input.IsActionJustPressed("ui_select"))
        {
            GD.Print("Mouse click");
            var treeTransform = CesInteraction.GetPointOnPlanet(this);
            if (treeTransform.HasValue)
                SpawnTree(treeTransform.Value);
        }

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
            var ray = GetMouseRayInLocalCoordinates();
            if (ray.HasValue)
            {
                var (origin, direction) = ray.Value;

                // Check for triangle intersection
                var hit = CheckRayTriangleIntersection(origin, direction);
                if (hit.HasValue)
                {
                    var (triangleIndex, distance) = hit.Value;

                    // Get triangle info to check if already divided
                    var triangleInfo = GetTriangleInfo(triangleIndex);
                    if (triangleInfo.HasValue)
                    {
                        if (triangleInfo.Value.IsDivided == 0)
                        {
                            // Mark triangle to be divided
                            MarkTriangleToDivide(triangleIndex);
                            GD.Print($"Divided 1 triangle (triangle {triangleIndex})");

                            // Apply the division immediately
                            ApplyManualDivisions();
                        }
                        else
                        {
                            GD.Print($"Triangle {triangleIndex} is already divided");
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

        _shiftMouseWasPressed = shiftMousePressed;

        // Ctrl key handling: temporarily increase triangle level by 6 and show info
        bool ctrlPressed = Input.IsKeyPressed(Key.Ctrl);

        if (ctrlPressed && !_ctrlWasPressed)
        {
            // Ctrl just pressed - find triangle and increase its level
            // Always use editor camera when Ctrl is pressed in debug mode
            var ray = GetMouseRayInLocalCoordinates();
            if (ray.HasValue)
            {
                var (origin, direction) = ray.Value;

                // Check for triangle intersection
                var hit = CheckRayTriangleIntersection(origin, direction);
                if (hit.HasValue)
                {
                    var (triangleIndex, distance) = hit.Value;

                    // Get current triangle info
                    var triangleInfo = GetTriangleInfo(triangleIndex);
                    if (triangleInfo.HasValue)
                    {
                        // Store the triangle index and original level
                        _ctrlTriangleIndex = triangleIndex;
                        _ctrlOriginalLevel = triangleInfo.Value.Level;

                        // Increase level by 6
                        SetTriangleLevel(triangleIndex, _ctrlOriginalLevel + 6);

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
                SetTriangleLevel(_ctrlTriangleIndex, _ctrlOriginalLevel);

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
            var ray = GetMouseRayInLocalCoordinates();
            if (ray.HasValue)
            {
                var (origin, direction) = ray.Value;

                // Check for triangle intersection
                var hit = CheckRayTriangleIntersection(origin, direction);
                if (hit.HasValue)
                {
                    var (triangleIndex, distance) = hit.Value;

                    // Get triangle info
                    var triangleInfo = GetTriangleInfo(triangleIndex);
                    if (triangleInfo.HasValue)
                    {
                        // Check if the triangle has a parent
                        int parentIndex = triangleInfo.Value.Parent;

                        if (parentIndex >= 0)
                        {
                            // Check if parent is divided
                            var parentInfo = GetTriangleInfo(parentIndex);
                            if (parentInfo.HasValue && parentInfo.Value.IsDivided == 1)
                            {
                                // Mark the parent triangle for merging
                                MarkTriangleToMerge(parentIndex);

                                // Apply the merge immediately
                                ApplyManualDivisions();
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

    private void SpawnTree(Transform3D pointTransform)
    {
        var tree = TreeModel.Instantiate() as Node3D;
        GetNode("Trees").AddChild(tree);
        tree.Owner = GetTree().EditedSceneRoot;
        tree.GlobalPosition = pointTransform.Origin;
        tree.GlobalTransform = pointTransform;
    }


    private void SimulateAsync()
    {
        // if (!Simulate || MainCamera == null) return;
        //
        // // Terminate running tasks
        // if ((_simTask == null || _simTask.IsCompleted) && (_genMeshTask == null || _genMeshTask.IsCompleted))
        // 	_simTask = Task.Run(() => SimulateStep());
        // else
        // {
        // 	if (ShowDebugMessages)
        // 		GD.Print("Simulation Thread busy");
        // }
    }

    private async Task SimulateStep()
    {
        if (ShowDebugMessages)
            GD.Print("Resuming simulation");
        await Task.Run(() =>
        {
            // Start timer
            var sw = new Stopwatch();
            sw.Start();

            Vector2[] simValue;

            _graphGenerator ??= new CesClient();
            _graphGenerator.gen = this;

            // if (_graphGenerator.State == null)
            //     return; //todo: enable again

            // _graphGenerator.SimulateStep(SimSpeed);

            // Get resutls 
            simValue = _graphGenerator.SimValue;

            if (_lowPolyLook && _mesh?.GetSurfaceCount() > 0)
            {
                var mesh = new ArrayMesh();
                var surfaceArray = new Array();
                surfaceArray.Resize((int)Mesh.ArrayType.Max);

                var uvs = simValue;
                var oldUvs = _mesh.SurfaceGetArrays(0)[(int)Mesh.ArrayType.TexUV];
                // Print a message  correct length of uvs
                // todo: check why this is happening and fix it
                if (uvs.Length != oldUvs.AsVector2Array().Length)
                {
                    if (ShowDebugMessages)
                        GD.Print("UVs length not equal to simValue length");
                    return;
                }

                // Convert Lists to arrays and assign to surface array
                surfaceArray[(int)Mesh.ArrayType.Vertex] = _mesh.SurfaceGetArrays(0)[(int)Mesh.ArrayType.Vertex];
                surfaceArray[(int)Mesh.ArrayType.TexUV] = uvs;
                surfaceArray[(int)Mesh.ArrayType.Normal] = _mesh.SurfaceGetArrays(0)[(int)Mesh.ArrayType.Normal];
                surfaceArray[(int)Mesh.ArrayType.Index] = _mesh.SurfaceGetArrays(0)[(int)Mesh.ArrayType.Index];

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
            }


            if (ShowDebugMessages)
                GD.Print($"Completed in {sw.ElapsedMilliseconds} ms");
        });
    }

    private void GenMeshAsync()
    {
        if (MainCamera == null)
            return;
        // GD.Print("Generating mesh");
        var camLocal = posToLocal(MainCamera.GlobalPosition);
        // Terminate running tasks
        if ((_simTask == null || _simTask.IsCompleted) && (_genMeshTask == null || _genMeshTask.IsCompleted))
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

    /// <summary>
    /// Gets the ray origin and direction from the global mouse position in local celestial coordinates.
    /// </summary>
    /// <returns>A tuple containing the ray origin and direction in local coordinates, or null if the camera is not available.</returns>
    public (Vector3 origin, Vector3 direction)? GetMouseRayInLocalCoordinates()
    {
        Camera3D camera = EditorInterface.Singleton.GetEditorViewport3D().GetCamera3D();

        // Use the camera's forward direction (-Z axis in camera's local space)
        var cameraTransform = camera.GlobalTransform;
        var globalOrigin = camera.GlobalPosition;
        var globalDirection = -cameraTransform.Basis.Z; // Camera looks down -Z axis

        // GD.Print($"Global Origin: {globalOrigin}");
        // GD.Print($"Global Direction: {globalDirection}");

        // Transform to local coordinates
        var localOrigin = posToLocal(globalOrigin);
        var localDirection = GlobalTransform.Basis.Inverse() * globalDirection;

        return (localOrigin, localDirection);
    }

    /// <summary>
    /// Checks if a ray intersects any triangle using a compute shader.
    /// </summary>
    public (int triangleIndex, float distance)? CheckRayTriangleIntersection(Vector3 rayOrigin, Vector3 rayDirection)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return null;
        }

        var state = _graphGenerator.State;

        // Debug: Print ray info
        // GD.Print($"Shader Input - Ray Origin: {rayOrigin}, Direction: {rayDirection}");
        // GD.Print($"Total Triangles: {state.nTris}");

        // Debug: Check masks
        var divMask = state.GetDividedMask();
        var deactMask = state.GetTDeactivatedMask();
        int visibleCount = 0;
        for (int i = 0; i < divMask.Length; i++)
        {
            if (divMask[i] == 0 && deactMask[i] == 0)
                visibleCount++;
        }
        GD.Print($"Visible triangles: {visibleCount} out of {state.nTris}");

        var shaderPath = "res://addons/celestial_sim/client/division/RayTriangleIntersection.slang";

        // Create output buffers
        var hitTriangleIndexBuffer = CesComputeUtils.CreateStorageBuffer(rd, new[] { -1 });
        var hitDistanceBuffer = CesComputeUtils.CreateStorageBuffer(rd, new[] { 1e30f });

        var bufferInfos = new BufferInfo[]
        {
            state.v_pos,                                                               // vertices
            state.t_abc,                                                               // triangle indices
            CesComputeUtils.CreateUniformBuffer(rd, rayOrigin),                       // ray origin
            CesComputeUtils.CreateUniformBuffer(rd, rayDirection),                    // ray direction
            hitTriangleIndexBuffer,                                                    // output hit triangle index
            hitDistanceBuffer,                                                         // output hit distance
            CesComputeUtils.CreateUniformBuffer(rd, state.nTris),                     // total triangles
            state.t_divided,                                                           // divided mask
            state.t_deactivated                                                        // deactivated mask
        };

        // Dispatch with only 1 thread since we're using a for loop in the shader
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Read results
        var hitIndex = CesComputeUtils.ConvertBufferToArray<int>(rd, hitTriangleIndexBuffer)[0];
        var hitDist = CesComputeUtils.ConvertBufferToArray<float>(rd, hitDistanceBuffer)[0];

        // Debug output
        GD.Print($"Shader Output - Hit Index: {hitIndex}, Distance: {hitDist}");

        // Clean up temporary buffers
        rd.FreeRid(hitTriangleIndexBuffer.buffer);
        rd.FreeRid(hitDistanceBuffer.buffer);

        if (hitIndex >= 0)
        {
            return (hitIndex, hitDist);
        }

        return null;
    }

    /// <summary>
    /// Increases the level of a specific triangle by the given amount using a compute shader.
    /// </summary>
    public void IncreaseTriangleLevel(int triangleIndex, int increaseAmount = 2)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = _graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/client/division/IncreaseLevelShader.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_lv,                                                                // triangle level buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(rd, triangleIndex),                   // triangle index to modify
            CesComputeUtils.CreateUniformBuffer(rd, increaseAmount)                   // amount to increase
        };

        // Dispatch with only 1 thread since we're modifying a single triangle
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffers (uniform buffers for index and amount)
        rd.FreeRid(bufferInfos[1].buffer);
        rd.FreeRid(bufferInfos[2].buffer);

        GD.Print($"Triangle {triangleIndex} level increased by {increaseAmount}");
    }

    /// <summary>
    /// Sets the level of a specific triangle to an exact value using a compute shader.
    /// </summary>
    public void SetTriangleLevel(int triangleIndex, int newLevel)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = _graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/client/division/SetTriangleLevelShader.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_lv,                                                                // triangle level buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(rd, triangleIndex),                   // triangle index to modify
            CesComputeUtils.CreateUniformBuffer(rd, newLevel)                         // new level value
        };

        // Dispatch with only 1 thread since we're modifying a single triangle
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffers (uniform buffers for index and level)
        rd.FreeRid(bufferInfos[1].buffer);
        rd.FreeRid(bufferInfos[2].buffer);
    }

    /// <summary>
    /// Marks a triangle to be divided on the next mesh update.
    /// </summary>
    public void MarkTriangleToDivide(int triangleIndex)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = _graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/client/division/MarkTriangleToDivide.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_to_divide_mask,                                                    // to-divide mask buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(rd, triangleIndex),                   // triangle index to mark
        };

        // Dispatch with only 1 thread
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffer
        rd.FreeRid(bufferInfos[1].buffer);
    }

    /// <summary>
    /// Marks a triangle to be merged on the next mesh update.
    /// The actual merge is performed by CesMergeSmallTris shader during UpdateTriangleGraph.
    /// </summary>
    public void MarkTriangleToMerge(int triangleIndex)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return;
        }

        var state = _graphGenerator.State;

        var shaderPath = "res://addons/celestial_sim/client/division/MarkTriangleToMerge.slang";

        var bufferInfos = new BufferInfo[]
        {
            state.t_to_merge_mask,                                                    // to-merge mask buffer (read-write)
            CesComputeUtils.CreateUniformBuffer(rd, triangleIndex),                   // triangle index to mark
        };

        // Dispatch with only 1 thread
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Clean up temporary buffer
        rd.FreeRid(bufferInfos[1].buffer);
    }

    /// <summary>
    /// Applies manual triangle divisions/undivisions and updates the mesh.
    /// Uses UpdateTriangleGraph with skipAutoDivisionMarking=true to preserve manual changes.
    /// </summary>
    private void ApplyManualDivisions()
    {
        if (_graphGenerator?.State == null || MainCamera == null)
        {
            GD.Print("Graph generator or camera not initialized");
            return;
        }

        var camLocal = posToLocal(MainCamera.GlobalPosition);

        // Update the triangle graph but skip automatic division marking
        // This will process the manually marked triangles without overriding them
        _graphGenerator.UpdateTriangleGraph(camLocal, skipAutoDivisionMarking: true);

        // Get results and update mesh
        var pos = _graphGenerator.Pos;
        var tris = _graphGenerator.Triangles;
        var norm = _graphGenerator.Norm;
        var simValue = _graphGenerator.SimValue;

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

        GD.Print("Manual division/undivision applied");
    }

    /// <summary>
    /// Applies debug triangle divisions from the DebugTriangleIndicesToDivide array.
    /// Called automatically after the first mesh generation when in debug mode.
    /// Positive indices divide triangles, negative indices merge triangles.
    /// </summary>
    private void ApplyDebugDivisions()
    {
        if (DebugTriangleIndicesToDivide.Count == 0)
            return;

        GD.Print($"Applying {DebugTriangleIndicesToDivide.Count} debug triangle operations...");

        int dividedCount = 0;
        int mergedCount = 0;

        // Process all triangle indices
        foreach (int value in DebugTriangleIndicesToDivide)
        {
            if (value >= 0)
            {
                // Positive index: divide triangle
                MarkTriangleToDivide(value);
                ApplyManualDivisions();
                dividedCount++;
            }
            else
            {
                // Negative index: merge triangle
                int triangleIndex = -value;

                // Check if triangle is divided before trying to merge
                var triangleInfo = GetTriangleInfo(triangleIndex);
                if (triangleInfo.HasValue && triangleInfo.Value.IsDivided == 1)
                {
                    MarkTriangleToMerge(triangleIndex);
                    ApplyManualDivisions();
                    mergedCount++;
                }
                else
                {
                    GD.Print($"Triangle {triangleIndex} cannot be merged (not divided or invalid)");
                }
            }
        }

        if (dividedCount > 0)
            GD.Print($"Auto-divided {dividedCount} triangle(s) in debug mode");
        if (mergedCount > 0)
            GD.Print($"Auto-merged {mergedCount} triangle(s) in debug mode (removed {mergedCount * 4} child triangles)");
    }

    /// <summary>
    /// Gets comprehensive information about a specific triangle using a compute shader.
    /// Returns vertex indices, level, flags (divided, deactivated, to_divide), 
    /// neighbours, icosphere index, and children triangles.
    /// </summary>
    public CesTriangleInfo? GetTriangleInfo(int triangleIndex)
    {
        if (_graphGenerator?.State == null)
        {
            GD.Print("Graph generator or state not initialized");
            return null;
        }

        var state = _graphGenerator.State;

        // Check if triangle index is valid
        if (triangleIndex < 0 || triangleIndex >= state.nTris)
        {
            GD.Print($"Invalid triangle index: {triangleIndex} (valid range: 0-{state.nTris - 1})");
            return null;
        }

        var shaderPath = "res://addons/celestial_sim/client/division/GetTriangleInfo.slang";

        // Create output buffer to hold the triangle info struct
        var outputBuffer = CesComputeUtils.CreateStorageBuffer(rd, new CesTriangleInfo[1]);

        var bufferInfos = new BufferInfo[]
        {
            state.t_abc,                                                               // triangle indices
            state.t_lv,                                                                // triangle level
            state.t_divided,                                                           // divided mask
            state.t_deactivated,                                                       // deactivated mask
            state.t_to_divide_mask,                                                    // to divide mask
            state.t_neight_ab,                                                         // neighbour AB
            state.t_neight_bc,                                                         // neighbour BC
            state.t_neight_ca,                                                         // neighbour CA
            state.t_ico_idx,                                                           // icosphere index
            state.t_a_t,                                                               // child A triangle
            state.t_b_t,                                                               // child B triangle
            state.t_c_t,                                                               // child C triangle
            state.t_center_t,                                                          // child center triangle
            state.t_parent,                                                            // parent triangle
            CesComputeUtils.CreateUniformBuffer(rd, triangleIndex),                   // triangle index to query
            outputBuffer                                                               // output buffer
        };

        // Dispatch with only 1 thread since we're querying a single triangle
        CesComputeUtils.DispatchShader(rd, shaderPath, bufferInfos, 1);

        // Read the result struct directly
        var results = CesComputeUtils.ConvertBufferToArray<CesTriangleInfo>(rd, outputBuffer);

        // Clean up temporary buffers
        rd.FreeRid(bufferInfos[15].buffer); // triangle index uniform buffer
        rd.FreeRid(outputBuffer.buffer);

        return results[0];
    }

    private void ConvertFloatArrayToVector3()
    {
        // All code is within this function

        // Step 1: Generate a sample float array
        // Let's create a float array with 9 elements (representing 3 Vector3 instances)
        var floatArray = new float[9];
        for (var i = 0; i < floatArray.Length; i++) floatArray[i] = i + 1; // Sample data: 1.0, 2.0, 3.0, etc.

        // Step 2: Ensure the length of the array is a multiple of 3
        if (floatArray.Length % 3 != 0)
            throw new ArgumentException("floatArray length must be a multiple of 3.");

        // Step 3: Convert the float array to a Span<float>
        var floatSpan = floatArray.AsSpan();

        // Step 4: Reinterpret the Span<float> as a Span<Vector3> using MemoryMarshal.Cast
        var vectorSpan = MemoryMarshal.Cast<float, Vector3>(floatSpan);

        // Step 5: Convert the Span<Vector3> to an array if needed
        var vectorArray = vectorSpan.ToArray();

        // Step 6: Print the results
        GD.Print("Converted Vector3 Array:");
        foreach (var vec in vectorArray) GD.Print(vec);
    }

    private async Task GenMesh(Vector3 localCameraPos)
    {
        await Task.Run(() =>
        {
            // ConvertFloatArrayToVector3();
            // Start timer
            var sw = new Stopwatch();
            sw.Start();

            Vector3[] pos;
            int[] tris;

            _graphGenerator ??= new CesClient();
            _graphGenerator.gen = this;

            // In debug mode, skip automatic division marking to preserve manual changes
            _graphGenerator.UpdateTriangleGraph(localCameraPos, skipAutoDivisionMarking: DebugMode);

            // Get resutls
            pos = _graphGenerator.Pos;
            tris = _graphGenerator.Triangles;
            var norm = _graphGenerator.Norm;
            var simValue = _graphGenerator.SimValue;

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
                CallDeferred(nameof(ApplyDebugDivisions));
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

    public void AddSimLayer(CesSimLayer cesSimLayerLayer, string name = null)
    {
        var layers_holder = GetNode("SimLayers");
        if (name != null) cesSimLayerLayer.Name = name;
        SimLayers.Add(cesSimLayerLayer);
        layers_holder.AddChild(cesSimLayerLayer);
        var sceneRoot = GetTree().EditedSceneRoot;
        if (sceneRoot != null) cesSimLayerLayer.Owner = sceneRoot;
        cesSimLayerLayer.celestial = this;
        ValuesUpdated = true;
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

    private void _UpdateSimLayers()
    {
        SimLayers.Clear();
        foreach (var layer in GetNode<Node3D>("SimLayers").GetChildren())
        {
            var lay = layer as CesSimLayer;
            lay.celestial = this;
            SimLayers.Add(lay);
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
        _UpdateSimLayers();
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
        _graphGenerator?.Dispose();
        _graphGenerator = null;
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
    public List<CesSimLayer> SimLayers { get; set; } = new();

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
    public bool PlaceTrees
    {
        get => _placeTrees;
        set
        {
            _placeTrees = value;
            ValuesUpdated = true;
        }
    }

    [Export]
    public PackedScene TreeModel;

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

    [Export]
    public bool VerifyIcosphereCorrectness
    {
        get => false;
        set
        {
            if (value)
            {
                RunVerification();
            }
        }
    }

    /// <summary>
    /// Verifies the correctness of the icosphere state, checking neighbor relationships
    /// and subdivision structure. Results are printed to the console.
    /// </summary>
    private void RunVerification()
    {
        if (_graphGenerator?.State == null)
        {
            GD.PrintErr("Cannot verify: Graph generator or state not initialized");
            return;
        }

        GD.Print("\n╔═══════════════════════════════════════════════════════╗");
        GD.Print("║     ICOSPHERE VERIFICATION TEST                       ║");
        GD.Print("╚═══════════════════════════════════════════════════════╝\n");

        var result = CesStateVerifier.VerifyComplete(_graphGenerator.State);

        // Print detailed results
        GD.Print(result.ToString());

        if (result.IsValid)
        {
            GD.Print("\n🎉 Icosphere verification PASSED! All checks successful.");
        }
        else
        {
            GD.PrintErr($"\n❌ Icosphere verification FAILED with {result.Errors.Count} error(s).");
            GD.PrintErr("See above for details.");
        }
    }


    // @formatter:on
}
