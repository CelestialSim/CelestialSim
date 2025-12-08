using Godot;


// Layers for creating terrain, they have a get height method
[Tool]
public abstract partial class CesLayer : Node3D
{
    protected CesState _cesState;
    // protected BufferInfo _VToUpdate;
    protected float _radius;
    public Vector3 celestialPos;

    public Vector3 craterCenterPos;

    // protected int _Level;
    // protected Vector3[] _NewVertices;
    public RenderingDevice rd;
    public int Seed;

    // This method is called on the first layer
    public void SetState(CesState s, float radius)
    {
        _cesState = s;
        // _VToUpdate = s.v_update_mask;
        rd = s.rd;
        _radius = radius;
    }

    // This method is called on following layers
    public void SetState(CesLayer l)
    {
        _cesState = l._cesState;
        // _VToUpdate = l._VToUpdate;
        _radius = l._radius;
        rd = l.rd;
        Seed = l.Seed;
        celestialPos = l.celestialPos;
        craterCenterPos = l.craterCenterPos;
    }

    // public void SetBackState()
    // {
    //     var bufferInfos = new BufferInfo[]
    //     {
    //         _cesState.v_pos,
    //         _VToUpdate,
    //         _cesState.t_abc  
    //     };
    //     CesComputeUtils.DispatchShader(rd "res://addons/celestial_sim/src/division/SetBackPos.slang", bufferInfos,
    //         _cesState.nVerts);
    // }

    public abstract void UpdatePos();

    // public void UpdateMesh()
    // {
    //     // GetNodeOrNull<PlanetGenerator>("%Planet_cs")?.UpdateMesh();
    // }
}
