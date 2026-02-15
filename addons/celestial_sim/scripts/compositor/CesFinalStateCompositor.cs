// Inspired by the code from Godot Forum post by user "axelsevens"
// Source: https://forum.godotengine.org/t/trying-to-draw-specific-meshes-to-texture-using-renderingdevice/63610/3
// Modified and integrated into CelestialSim project

// Compositor effect that renders CesState geometry in scene space
#nullable enable
using System;
using Godot;
using Godot.Collections;

[Tool]
[GlobalClass]
public partial class CesFinalStateCompositor : CompositorEffect
{
    private RenderingDevice? rd;
    private const string DefaultDrawShaderPath = "res://addons/celestial_sim/scripts/compositor/ces_final_state_compositor.glsl";

    [Export]
    public NodePath? TargetPath { get; set; }

    [Export]
    public Vector3 TargetPosition { get; set; } = Vector3.Zero;

    [Export]
    public float Radius { get; set; } = 1.0f;

    [Export]
    public float AmbientStrength { get; set; } = 0.05f;

    [Export]
    public Color SphereColor { get; set; } = new Color(0, 0, 0, 1);

    private RDShaderFile? _drawShaderFile;
    private Rid drawShader;

    private Rid vertexStorageBuffer;
    private Rid triangleStorageBuffer;
    private Rid structuredVertexArray;
    private Rid structuredUniformSet;
    private uint structuredVertexCount;
    private CesState? cesState;

    private Rid renderPipeline;
    private long framebufferFormat = -1;
    private long vertexFormat;

    public CesFinalStateCompositor() : base()
    {
        EffectCallbackType = EffectCallbackTypeEnum.PostTransparent;
        AccessResolvedColor = true;
        AccessResolvedDepth = true;
        RenderingServer.CallOnRenderThread(Callable.From(Construct));
    }

    private void Construct()
    {
        rd = RenderingServer.GetRenderingDevice();
        if (rd is null) return;

        _drawShaderFile = ResourceLoader.Load<RDShaderFile>(
            DefaultDrawShaderPath,
            cacheMode: ResourceLoader.CacheMode.Ignore
        );
        if (_drawShaderFile is null) return;

        var spirv = _drawShaderFile.GetSpirV();
        if (!string.IsNullOrEmpty(spirv.CompileErrorVertex))
        {
            GD.PrintErr($"CesFinalStateCompositor: Vertex shader compile error: {spirv.CompileErrorVertex}");
            return;
        }
        if (!string.IsNullOrEmpty(spirv.CompileErrorFragment))
        {
            GD.PrintErr($"CesFinalStateCompositor: Fragment shader compile error: {spirv.CompileErrorFragment}");
            return;
        }

        drawShader = rd.ShaderCreateFromSpirV(spirv);
        if (!drawShader.IsValid) return;

        vertexFormat = rd.VertexFormatCreate([]);

        cesState = CesCoreState.CreateCoreState(rd);
        vertexStorageBuffer = cesState.v_pos.buffer;
        triangleStorageBuffer = cesState.t_abc.buffer;
        structuredVertexCount = cesState.nTris * 3;

        structuredVertexArray = rd.VertexArrayCreate(structuredVertexCount, vertexFormat, []);

        RDUniform vertexUniform = new()
        {
            Binding = 0,
            UniformType = RenderingDevice.UniformType.StorageBuffer,
        };
        vertexUniform.AddId(vertexStorageBuffer);

        RDUniform triUniform = new()
        {
            Binding = 1,
            UniformType = RenderingDevice.UniformType.StorageBuffer,
        };
        triUniform.AddId(triangleStorageBuffer);

        structuredUniformSet = rd.UniformSetCreate([vertexUniform, triUniform], drawShader, 0);
    }

    private bool EnsurePipeline(long fbFormat)
    {
        if (rd is null || !drawShader.IsValid) return false;
        if (framebufferFormat == fbFormat && renderPipeline.IsValid) return true;

        if (renderPipeline.IsValid) rd.FreeRid(renderPipeline);
        framebufferFormat = fbFormat;

        RDPipelineColorBlendState blend = new();
        blend.Attachments.Add(new RDPipelineColorBlendStateAttachment());

        RDPipelineDepthStencilState depthState = new()
        {
            EnableDepthTest = true,
            EnableDepthWrite = true,
            DepthCompareOperator = RenderingDevice.CompareOperator.GreaterOrEqual,
        };

        RDPipelineRasterizationState rasterState = new()
        {
            CullMode = RenderingDevice.PolygonCullMode.Back,
            FrontFace = RenderingDevice.PolygonFrontFace.Clockwise,
        };

        renderPipeline = rd.RenderPipelineCreate(
            drawShader,
            fbFormat,
            vertexFormat,
            RenderingDevice.RenderPrimitive.Triangles,
            rasterState,
            new RDPipelineMultisampleState(),
            depthState,
            blend
        );

        return renderPipeline.IsValid;
    }

    public override void _RenderCallback(int effectCallbackType, RenderData renderData)
    {
        if (rd is null || !drawShader.IsValid) return;

        RenderSceneBuffersRD? sceneBuffers = renderData.GetRenderSceneBuffers() as RenderSceneBuffersRD;
        if (sceneBuffers == null) return;

        Vector2I size = sceneBuffers.GetInternalSize();
        if (size.X == 0 || size.Y == 0) return;

        Rid colorImage = sceneBuffers.GetColorLayer(0);
        Rid depthImage = sceneBuffers.GetDepthLayer(0);
        if (!colorImage.IsValid || !depthImage.IsValid) return;

        Rid framebuffer = rd.FramebufferCreate([colorImage, depthImage]);
        if (!framebuffer.IsValid) return;

        long fbFormat = rd.FramebufferGetFormat(framebuffer);
        if (!EnsurePipeline(fbFormat))
        {
            rd.FreeRid(framebuffer);
            return;
        }

        RenderSceneDataRD? sceneData = renderData.GetRenderSceneData() as RenderSceneDataRD;
        if (sceneData == null)
        {
            rd.FreeRid(framebuffer);
            return;
        }

        Transform3D camTransform = sceneData.GetCamTransform();
        Projection camProjection = sceneData.GetCamProjection();
        Transform3D viewTransform = camTransform.AffineInverse();

        Vector3 position = TargetPosition;
        if (TargetPath != null && !TargetPath.IsEmpty)
        {
            SceneTree? tree = Engine.GetMainLoop() as SceneTree;
            Node? root = tree?.Root;
            Node3D? targetNode = root?.GetNodeOrNull<Node3D>(TargetPath);
            if (targetNode != null)
            {
                position = targetNode.GlobalTransform.Origin;
            }
        }

        Transform3D modelTransform = new Transform3D(Basis.Identity.Scaled(Vector3.One * Radius), position);

        DirectionalLight3D? sceneLight = FindDirectionalLight();
        Vector3 lightDir = sceneLight != null
            ? -sceneLight.GlobalTransform.Basis.Z.Normalized()
            : new Vector3(0, 0, 1);
        Color lightColor = sceneLight != null ? sceneLight.LightColor : new Color(1, 1, 1, 1);
        float lightIntensity = sceneLight != null ? sceneLight.LightEnergy : 1.0f;

        Vector3 lightDirView = (viewTransform.Basis * lightDir).Normalized();

        float[] pushData = new float[60];
        WriteTransformToArray(modelTransform, pushData, 0);
        WriteTransformToArray(viewTransform, pushData, 16);
        WriteProjectionToArray(camProjection, pushData, 32);
        pushData[48] = lightDirView.X;
        pushData[49] = lightDirView.Y;
        pushData[50] = lightDirView.Z;
        pushData[51] = lightIntensity;
        pushData[52] = lightColor.R;
        pushData[53] = lightColor.G;
        pushData[54] = lightColor.B;
        pushData[55] = AmbientStrength;
        pushData[56] = SphereColor.R;
        pushData[57] = SphereColor.G;
        pushData[58] = SphereColor.B;
        pushData[59] = SphereColor.A;

        byte[] pushBytes = new byte[pushData.Length * sizeof(float)];
        Buffer.BlockCopy(pushData, 0, pushBytes, 0, pushBytes.Length);

        long drawList = rd.DrawListBegin(framebuffer, default(RenderingDevice.DrawFlags), [], 1.0f, 0);
        if (drawList < 0)
        {
            rd.FreeRid(framebuffer);
            return;
        }

        rd.DrawListBindRenderPipeline(drawList, renderPipeline);
        rd.DrawListBindVertexArray(drawList, structuredVertexArray);
        rd.DrawListBindUniformSet(drawList, structuredUniformSet, 0);
        rd.DrawListSetPushConstant(drawList, pushBytes, (uint)pushBytes.Length);
        rd.DrawListDraw(drawList, false, 1);
        rd.DrawListEnd();

        rd.FreeRid(framebuffer);
    }

    private static void WriteTransformToArray(Transform3D transform, float[] array, int offset)
    {
        array[offset + 0] = transform.Basis.X.X;
        array[offset + 1] = transform.Basis.X.Y;
        array[offset + 2] = transform.Basis.X.Z;
        array[offset + 3] = 0;
        array[offset + 4] = transform.Basis.Y.X;
        array[offset + 5] = transform.Basis.Y.Y;
        array[offset + 6] = transform.Basis.Y.Z;
        array[offset + 7] = 0;
        array[offset + 8] = transform.Basis.Z.X;
        array[offset + 9] = transform.Basis.Z.Y;
        array[offset + 10] = transform.Basis.Z.Z;
        array[offset + 11] = 0;
        array[offset + 12] = transform.Origin.X;
        array[offset + 13] = transform.Origin.Y;
        array[offset + 14] = transform.Origin.Z;
        array[offset + 15] = 1;
    }

    private static void WriteProjectionToArray(Projection projection, float[] array, int offset)
    {
        array[offset + 0] = projection.X.X;
        array[offset + 1] = projection.X.Y;
        array[offset + 2] = projection.X.Z;
        array[offset + 3] = projection.X.W;
        array[offset + 4] = projection.Y.X;
        array[offset + 5] = projection.Y.Y;
        array[offset + 6] = projection.Y.Z;
        array[offset + 7] = projection.Y.W;
        array[offset + 8] = projection.Z.X;
        array[offset + 9] = projection.Z.Y;
        array[offset + 10] = projection.Z.Z;
        array[offset + 11] = projection.Z.W;
        array[offset + 12] = projection.W.X;
        array[offset + 13] = projection.W.Y;
        array[offset + 14] = projection.W.Z;
        array[offset + 15] = projection.W.W;
    }

    private static DirectionalLight3D? FindDirectionalLight()
    {
        SceneTree? tree = Engine.GetMainLoop() as SceneTree;
        Node? root = tree?.Root;
        if (root == null) return null;

        return FindDirectionalLightRecursive(root);
    }

    private static DirectionalLight3D? FindDirectionalLightRecursive(Node node)
    {
        if (node is DirectionalLight3D light) return light;

        foreach (Node child in node.GetChildren())
        {
            DirectionalLight3D? found = FindDirectionalLightRecursive(child);
            if (found != null) return found;
        }

        return null;
    }

    public override void _Notification(int what)
    {
        if (what == NotificationPredelete)
        {
            if (rd != null)
            {
                if (vertexStorageBuffer.IsValid) rd.FreeRid(vertexStorageBuffer);
                if (triangleStorageBuffer.IsValid) rd.FreeRid(triangleStorageBuffer);
                if (structuredVertexArray.IsValid) rd.FreeRid(structuredVertexArray);
                if (renderPipeline.IsValid) rd.FreeRid(renderPipeline);
                if (drawShader.IsValid) rd.FreeRid(drawShader);
            }
        }
    }
}
