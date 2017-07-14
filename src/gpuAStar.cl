// GPU A* program

__kernel void gpuAStar(__global const float4 *nodes,
                                const uint    nodesSize,
                       __global const float4 *edges,
                                const uint    edgesSize,
                       __global const uint2  *adjacencyMap,
                                const uint    adjacencyMapSize,
                       __global       float4 *paths,
                                const uint    pathsSize)
{

}
