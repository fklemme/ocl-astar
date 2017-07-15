// GPU A* program

void push(__local float2 *heap, size_t *size, float key, float value) {}

__kernel void gpuAStar(__global const float4 *nodes, // id, x, y, unused
                                const ulong   nodesSize,
                       __global const float4 *edges, // source, destination, cost, unused
                                const ulong   edgesSize,
                       __global const uint2  *adjacencyMap, // edges_begin, edges_end
                                const ulong   adjacencyMapSize,
                                const ulong   numberOfAgents,
                       __global const uint2  *srcDstList, // source id, destination id
                       __global       uint2  *paths, // offset = GID * maxPathLength
                                const ulong   maxPathLength,
                       __local        float2 *open)
{
    const size_t GID = get_global_id(0);
    size_t openSize = 0;

    if (GID >= numberOfAgents)
        return;

    const uint source      = srcDstList[GID].x;
    const uint destination = srcDstList[GID].y;

    push(open, &openSize, source, 0.0f);
}
