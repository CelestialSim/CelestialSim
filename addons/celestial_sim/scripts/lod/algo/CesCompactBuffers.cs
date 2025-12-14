using System;
using Godot;


public class CesCompactBuffers
{
    private readonly RenderingDevice _rd;
    private readonly string compactTrisShaderPath = "res://addons/celestial_sim/scripts/lod/algo/CompactTris.slang";
    private readonly string markActiveVertsPath = "res://addons/celestial_sim/scripts/lod/algo/MarkActiveVertices.slang";
    private readonly string compactVertsShaderPath = "res://addons/celestial_sim/scripts/lod/algo/CompactVertices.slang";
    private readonly string remapTrisVertsPath = "res://addons/celestial_sim/scripts/lod/algo/RemapTriangleVertices.slang";

    public CesCompactBuffers(RenderingDevice rd)
    {
        _rd = rd;
    }

    public uint Compact(CesState state)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        // Build active prefix on CPU (1 for active, 0 for deactivated)
        var deactivated = state.GetTDeactivatedMask().ToArray();
        CesComputeUtils.SumArrayInPlace(deactivated.AsSpan(), invert: true);

        var activeCount = deactivated.Length == 0 ? 0u : (uint)deactivated[^1];
        if (activeCount == 0)
        {
            return 0;
        }

        if (activeCount == state.nTris)
        {
            return 0;
        }

        var activePrefixBuffer = CesComputeUtils.CreateStorageBuffer(_rd, deactivated);

        uint dstCount = activeCount;
        uint bytesInt = (uint)sizeof(int);
        uint bytesTri = bytesInt * 4u;

        // Allocate destination buffers sized to the active triangle count
        var tAbcDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesTri);
        var tDividedDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tLvDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tToDivDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tToMergeDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tIcoIdxDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tNeighAbDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tNeighBcDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tNeighCaDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tATDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tBTDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tCTDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tCenterTDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tParentDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var tDeactivatedDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, dstCount * bytesInt);
        var bufferInfos = new BufferInfo[]
        {
            state.t_deactivated,      // 0
            activePrefixBuffer,       // 1
            state.t_abc,              // 2
            state.t_divided,          // 3
            state.t_lv,               // 4
            state.t_to_divide_mask,   // 5
            state.t_to_merge_mask,    // 6
            state.t_ico_idx,          // 7
            state.t_neight_ab,        // 8
            state.t_neight_bc,        // 9
            state.t_neight_ca,        // 10
            state.t_a_t,              // 11
            state.t_b_t,              // 12
            state.t_c_t,              // 13
            state.t_center_t,         // 14
            state.t_parent,           // 15
            tAbcDst,                  // 16
            tDividedDst,              // 17
            tLvDst,                   // 18
            tToDivDst,                // 19
            tToMergeDst,              // 20
            tIcoIdxDst,               // 21
            tNeighAbDst,              // 22
            tNeighBcDst,              // 23
            tNeighCaDst,              // 24
            tATDst,                   // 25
            tBTDst,                   // 26
            tCTDst,                   // 27
            tCenterTDst,              // 28
            tParentDst,               // 29
            tDeactivatedDst,          // 30
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris), // 31
            CesComputeUtils.CreateUniformBuffer(_rd, activeCount)  // 32
        };

        CesComputeUtils.DispatchShader(_rd, compactTrisShaderPath, bufferInfos, state.nTris);

        // Swap in compacted buffers
        var oldAbc = state.t_abc;
        var oldDivided = state.t_divided;
        var oldLv = state.t_lv;
        var oldToDiv = state.t_to_divide_mask;
        var oldToMerge = state.t_to_merge_mask;
        var oldIcoIdx = state.t_ico_idx;
        var oldNeighAb = state.t_neight_ab;
        var oldNeighBc = state.t_neight_bc;
        var oldNeighCa = state.t_neight_ca;
        var oldAT = state.t_a_t;
        var oldBT = state.t_b_t;
        var oldCT = state.t_c_t;
        var oldCenterT = state.t_center_t;
        var oldParent = state.t_parent;
        var oldDeactivated = state.t_deactivated;

        state.t_abc = tAbcDst;
        state.t_divided = tDividedDst;
        state.t_lv = tLvDst;
        state.t_to_divide_mask = tToDivDst;
        state.t_to_merge_mask = tToMergeDst;
        state.t_ico_idx = tIcoIdxDst;
        state.t_neight_ab = tNeighAbDst;
        state.t_neight_bc = tNeighBcDst;
        state.t_neight_ca = tNeighCaDst;
        state.t_a_t = tATDst;
        state.t_b_t = tBTDst;
        state.t_c_t = tCTDst;
        state.t_center_t = tCenterTDst;
        state.t_parent = tParentDst;
        state.t_deactivated = tDeactivatedDst;

        state.nTris = activeCount;
        state.nDeactivatedTris = 0;

        void FreeBuffer(Rid rid)
        {
            if (rid.Equals(default))
            {
                return;
            }

            RenderingServer.CallOnRenderThread(Callable.From(() => _rd.FreeRid(rid)));
        }

        // Free old buffers now that the compacted ones are active
        FreeBuffer(oldAbc.buffer);
        FreeBuffer(oldDivided.buffer);
        FreeBuffer(oldLv.buffer);
        FreeBuffer(oldToDiv.buffer);
        FreeBuffer(oldToMerge.buffer);
        FreeBuffer(oldIcoIdx.buffer);
        FreeBuffer(oldNeighAb.buffer);
        FreeBuffer(oldNeighBc.buffer);
        FreeBuffer(oldNeighCa.buffer);
        FreeBuffer(oldAT.buffer);
        FreeBuffer(oldBT.buffer);
        FreeBuffer(oldCT.buffer);
        FreeBuffer(oldCenterT.buffer);
        FreeBuffer(oldParent.buffer);
        FreeBuffer(oldDeactivated.buffer);
        FreeBuffer(activePrefixBuffer.buffer);

        // ---------------- Vertex compaction ----------------
        if (state.nVerts == 0)
        {
            return activeCount;
        }

        var vertActiveMask = CesComputeUtils.CreateEmptyStorageBuffer(_rd, state.nVerts * bytesInt);
        RenderingServer.CallOnRenderThread(Callable.From(() => _rd.BufferClear(vertActiveMask.buffer, 0, vertActiveMask.filledSize)));

        var markVertBuffers = new BufferInfo[]
        {
            state.t_abc,            // 0
            state.t_deactivated,    // 1
            vertActiveMask,         // 2
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris), // 3
            CesComputeUtils.CreateUniformBuffer(_rd, state.nVerts) // 4
        };

        CesComputeUtils.DispatchShader(_rd, markActiveVertsPath, markVertBuffers, state.nTris);

        var vertActive = CesComputeUtils.ConvertBufferToArray<int>(_rd, vertActiveMask).ToArray();
        CesComputeUtils.SumArrayInPlace(vertActive.AsSpan());
        var activeVertsCount = vertActive.Length == 0 ? 0u : (uint)vertActive[^1];

        if (activeVertsCount == 0)
        {
            FreeBuffer(vertActiveMask.buffer);
            return activeCount;
        }

        if (activeVertsCount == state.nVerts)
        {
            FreeBuffer(vertActiveMask.buffer);
            return activeCount;
        }

        var paddedVertPrefix = vertActive.Length < 4 ? new int[4] : vertActive;
        if (!ReferenceEquals(paddedVertPrefix, vertActive))
        {
            Array.Copy(vertActive, paddedVertPrefix, vertActive.Length);
        }

        var vertPrefixBuffer = CesComputeUtils.CreateStorageBuffer(_rd, paddedVertPrefix);

        var vPosBytes = activeVertsCount * 4u * (uint)sizeof(float);
        var vPosDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, vPosBytes);
        var vUpdateMaskDst = CesComputeUtils.CreateEmptyStorageBuffer(_rd, activeVertsCount * bytesInt);

        var compactVertBuffers = new BufferInfo[]
        {
            vertActiveMask,               // 0
            vertPrefixBuffer,             // 1
            state.v_pos,                  // 2
            state.v_update_mask,          // 3
            vPosDst,                      // 4
            vUpdateMaskDst,               // 5
            CesComputeUtils.CreateUniformBuffer(_rd, state.nVerts),       // 6
            CesComputeUtils.CreateUniformBuffer(_rd, activeVertsCount)    // 7
        };

        CesComputeUtils.DispatchShader(_rd, compactVertsShaderPath, compactVertBuffers, state.nVerts);

        var oldVPos = state.v_pos;
        var oldVUpdate = state.v_update_mask;

        state.v_pos = vPosDst;
        state.v_update_mask = vUpdateMaskDst;
        state.nVerts = activeVertsCount;

        // Remap triangle vertex indices to compacted vertex indices
        var tAbcRemapped = CesComputeUtils.CreateEmptyStorageBuffer(_rd, state.nTris * bytesTri);
        var remapBuffers = new BufferInfo[]
        {
            state.t_abc,          // 0 src
            vertPrefixBuffer,     // 1 prefix
            tAbcRemapped,         // 2 dst
            CesComputeUtils.CreateUniformBuffer(_rd, state.nTris),        // 3
            CesComputeUtils.CreateUniformBuffer(_rd, activeVertsCount)    // 4
        };

        CesComputeUtils.DispatchShader(_rd, remapTrisVertsPath, remapBuffers, state.nTris);

        var oldAbcRemapSrc = state.t_abc;
        state.t_abc = tAbcRemapped;

        FreeBuffer(oldVPos.buffer);
        FreeBuffer(oldVUpdate.buffer);
        FreeBuffer(vertActiveMask.buffer);
        FreeBuffer(vertPrefixBuffer.buffer);
        FreeBuffer(oldAbcRemapSrc.buffer);

        return activeCount;
    }
}
