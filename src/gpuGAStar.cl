// GPU GA* program

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define SQRT2 1.41421356237f

// ----- Types ----------------------------------------------------------------
typedef struct {
    uint  first;
    float second;
} uint_float;

typedef struct {  // Depending on use case...
    uint  closed; // only one of the first two members is used here
    uint  node;   // the other one gives padding for memory alignment
    float totalCost;
    uint  predecessor;
} Info;

// ----- Helper ---------------------------------------------------------------
uint_float _read_heap(__global uint_float *open, size_t index) {
    return open[index];
}

void _write_heap(__global uint_float *open, size_t index, uint_float value) {
    open[index] = value;
}

// ----- OpenList functions ---------------------------------------------------
uint top(__global uint_float *open) {
    return open[0].first;
}

void _push_impl(__global uint_float *open, size_t *size, uint value, float cost) {
    size_t index = (*size)++;

    while (index > 0) {
        size_t parent = (index - 1) / 2;

        uint_float pValue = _read_heap(open, parent);
        if (cost < pValue.second) {
            _write_heap(open, index, pValue);
            index = parent;
        } else
            break;
    }

    _write_heap(open, index, (uint_float){value, cost});
}

void push(__global uint_float *open, size_t *size, uint value, float cost) {
    _push_impl(open, size, value, cost);
}

void update(__global uint_float *open, size_t index, uint value, float cost) {
    _push_impl(open, &index, value, cost);
}

void pop(__global uint_float *open, size_t *size) {
    uint_float value = _read_heap(open, --(*size));
    size_t     index = 0;

    while (index < *size / 2) {
        size_t child = index * 2 + 1;

        uint_float cValue = _read_heap(open, child);
        if (child + 1 < *size) {
            uint_float c1Value = _read_heap(open, child + 1);

            if (c1Value.second < cValue.second) {
                ++child;
                cValue = c1Value;
            }
        }

        if (cValue.second < value.second) {
            _write_heap(open, index, cValue);
            index = child;
        } else
            break;
    }

    _write_heap(open, index, value);
}

uint find(__global uint_float *open, size_t *size, uint value) {
    for (uint index = 0; index < *size; ++index) {
        uint_float iValue = _read_heap(open, index);
        if (iValue.first == value)
            return index;
    }

    return *size;
}

// ----- Kernels --------------------------------------------------------------
__kernel void clearList(__global uint *list, const ulong size) {
    if (get_global_id(0) < size)
        list[get_global_id(0)] = 0;
}

__kernel void extractAndExpand(__global const uint_float *edges,            // destination index, stepCost
                                        const ulong       edgesSize,
                               __global const uint2      *adjacencyMap,     // edges_begin, edges_end
                                        const ulong       adjacencyMapSize,
                                        const ulong       numberOfQueues,   // provides offset ...
                                        const ulong       sizeOfAQueue,     // provides offset ...
                                        const uint        destination,      // destination index
                               __global       uint_float *openLists,        // aka "Q" priority queues
                               __global       uint       *openSizes,
                               __global       Info       *info,             // closed list, see members at the top
                               __global       Info       *slistChunks,      // "S" list, divided into chunks
                               __global       uint       *slistSizes,
                                        const ulong       slistChunkSize,
                               __global       uint       *returnCode)
{
    // Parallel for each queue (one dimensional)
    const size_t GID = get_global_id(0);

    if (GID >= numberOfQueues)
        return;

    __global uint_float *openList = openLists + GID * sizeOfAQueue;
    __global Info       *slist = slistChunks + GID * slistChunkSize;

    size_t openSize = openSizes[GID]; // read open list size
    uint   slistSize = 0;

    if (openSize == 0) {
        atomic_min(returnCode, 2);
        return; // failure: no path found!
    }

    const uint current = top(openList);
    pop(openList, &openSize);

    if (current == destination) {
        atomic_min(returnCode, 0);
        return; // success: path found!
    }

    // In this algorithm, "closed" means already added to open list.
    // --> nothing to do here.
    const Info currentInfo = info[current];

    const uint2 edgeRange = adjacencyMap[current];
    for (uint edge = edgeRange.x; edge != edgeRange.y; ++edge) {
        const uint  nbNode     = edges[edge].first;
        const float nbStepCost = edges[edge].second;

        const float nbTotalCost = currentInfo.totalCost + nbStepCost;
        slist[slistSize++] = (Info){0, nbNode, nbTotalCost, current};
    }

    // Write back new list sizes
    openSizes[GID] = (uint) openSize;
    slistSizes[GID] = slistSize;

    atomic_min(returnCode, 1); // still running...
}

__kernel void duplicateDetection(         const ulong       numberOfQueues,   // provides offset ...
                                 __global       Info       *info,             // closed list, see members at the top
                                 __global       Info       *slistChunks,      // "S" list, divided into chunks
                                 __global       uint       *slistSizes,
                                          const ulong       slistChunkSize,   // equals "tlistChunkSize" as well
                                 __global       Info       *tlistChunks,      // "T" list, divided into chunks
                                 __global       uint       *tlistSizes)
{
    // Parallel for each element in S-list (two dimensional)
    const uint2 GID = {get_global_id(0), get_global_id(1)};

    if (GID.x >= numberOfQueues || GID.y >= slistChunkSize)
        return;

    __global Info *slist = slistChunks + GID.x * slistChunkSize;
    uint slistSize = slistSizes[GID.x];

    if (GID.y >= slistSize)
        return;

    const Info current = slist[GID.y];
    const Info nodeInfo = info[current.node];

    // In this algorithm, "closed" means already added to open list.
    if (nodeInfo.closed == 1 && nodeInfo.totalCost < current.totalCost)
        return; // better candidate already in open list

    // TODO: Dedublication with hashing and stuff...

    __global Info *tlist = tlistChunks + GID.x * slistChunkSize;
    const uint index = atomic_inc(tlistSizes + GID.x);
    tlist[index] = current;
}

// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
float heuristic(int2 source, int2 destination) {
    const int dx = abs(destination.x - source.x);
    const int dy = abs(destination.y - source.y);
    return (dx + dy) + (SQRT2 - 2) * min(dx, dy);
}

__kernel void computeAndPushBack(__global const int2       *nodes,            // x, y
                                          const ulong       nodesSize,
                                          const ulong       numberOfQueues,   // provides offset ...
                                          const ulong       sizeOfAQueue,     // provides offset ...
                                          const uint        destination,      // destination index
                                 __global       uint_float *openLists,        // aka "Q" priority queues
                                 __global       uint       *openSizes,
                                 __global       Info       *info,             // closed list, see members at the top
                                 __global       Info       *tlistChunks,      // "T" list, divided into chunks
                                 __global       uint       *tlistSizes,
                                          const ulong       tlistChunkSize)
{
    // Parallel for each queue (one dimensional)
    const size_t GID = get_global_id(0);

    if (GID >= numberOfQueues)
        return;

    const int2 destNode = nodes[destination];

    __global uint_float *openList = openLists + GID * sizeOfAQueue;
    size_t openSize = openSizes[GID]; // read open list size

    for (size_t i = 0; i < tlistChunkSize; ++i) {
        const size_t gindex = i * numberOfQueues + GID;
        const size_t chunkIndex = gindex / tlistChunkSize;
        const size_t indexInChunk = gindex % tlistChunkSize;

        const uint chunkSize = tlistSizes[chunkIndex];

        if (indexInChunk >= chunkSize)
            continue;

        const Info current = tlistChunks[gindex];

        // In this algorithm, "closed" means already added to open list.
        info[current.node].closed = 1; // close node

        // Update totalCost and predecessor as one 64 bit transaction.
        ulong *currentCostPred = (ulong *) &current.totalCost;
        __global ulong *infoCostPred = (__global ulong *) &info[current.node].totalCost;
        ulong oldCostPred = atom_xchg(infoCostPred, *currentCostPred);
        float *oldCost = (float *) &oldCostPred;

        // assert: current.totalCost > 0.0f
        while (*oldCost != 0.0f && *oldCost < current.totalCost) {
            // The old entry was better. Swap back!
            oldCostPred = atom_xchg(infoCostPred, *currentCostPred);
        }

        float h = heuristic(nodes[current.node], destNode);
        push(openList, &openSize, current.node, current.totalCost + h);
    }

    // Write back new list size
    openSizes[GID] = (uint) openSize;
}
