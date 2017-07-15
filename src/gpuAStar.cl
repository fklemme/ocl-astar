// GPU A* program

__kernel void gpuAStar(__global const float4 *nodes, // id, x, y, unused
                                const uint    nodesSize,
                       __global const float4 *edges, // source, destination, cost, unused
                                const uint    edgesSize,
                       __global const uint2  *adjacencyMap, // edges_begin, edges_end
                                const uint    adjacencyMapSize,
                                const uint    numberOfAgents,
                       __global const uint4  *srcDstList,
                       __global       uint2  *paths, // offset = GID * maxPathLength
                                const uint    maxPathLength)
{
    const size_t GID = get_global_id(0);

    if (GID >= numberOfAgents)
        return;

    // FIXME: Most stupid pathfinding possible:

    const uint2 source      = srcDstList[GID].xy;
    const uint2 destination = srcDstList[GID].zw;
    uint2 node = source;

    size_t nodeIndex = GID * maxPathLength;
    paths[nodeIndex++] = source;

    while (node.x != destination.x || node.y != destination.y) {
        int dx = (int) destination.x - (int) node.x;
        int dy = (int) destination.y - (int) node.y;

        if (abs(dx) > abs(dy)) {
            node.x += dx < 0 ? -1 : 1;
        } else {
            node.y += dy < 0 ? -1 : 1;
        }

        paths[nodeIndex++] = node;
    }
}
