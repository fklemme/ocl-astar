// GPU GA* program

// ----- Types ----------------------------------------------------------------
typedef struct {
    uint  first;
    float second;
} uint_float;

typedef struct {
    uint  closed;
    float totalCost;
    uint  predecessor;
    uint  _reserved; // for memory alignment
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
                               __global       uint_float *slistChunks,      // "S" list, divided into chunks
                               __global       uint       *slistSizes,
                                        const ulong       slistChunkSize)
{
    // Parallel for each queue (one dimensional)
    const size_t GID = get_global_id(0);

    if (GID >= numberOfQueues)
        return;

    __global uint_float *openList = openLists + GID * sizeOfAQueue;
    __global uint_float *slist = slistChunks + GID * slistChunkSize;

    size_t openSize = openSizes[GID]; // read open list size
    size_t slistSize = 0;

    if (openSize == 0)
        return;

    const uint current = top(openList);
    pop(openList, &openSize);

    if (current == destination)
        return; // TODO: target found

    info[current].closed = 1; // close node
    const float totalCost = info[current].totalCost;

    const uint2 edgeRange = adjacencyMap[current];
    for (uint edge = edgeRange.x; edge != edgeRange.y; ++edge) {
        const uint  nbNode     = edges[edge].first;
        const float nbStepCost = edges[edge].second;

        const float nbTotalCost = totalCost + nbStepCost;
        slist[slistSize++] = (uint_float){nbNode, nbTotalCost};
    }

    // Write back new list sizes.
    openSizes[GID] = (uint) openSize;
    slistSizes[GID] = (uint) slistSize;
}

__kernel void checkAndFinalize()
{
}

__kernel void duplicateDetection(         const ulong       numberOfQueues,   // provides offset ...
                                          const ulong       sizeOfAQueue,     // provides offset ...
                                 __global       Info       *info,             // closed list, see members at the top
                                 __global       uint_float *slistChunks,      // "S" list, divided into chunks
                                 __global       uint       *slistSizes,
                                          const ulong       slistChunkSize,   // equals "tlistChunkSize" as well
                                 __global       uint_float *tlistChunks,      // "T" list, divided into chunks
                                 __global       uint       *tlistSizes)
{
    // Parallel for each element in S-list (two dimensional)
    const size_t GID[2] = {get_global_id(0),
                           get_global_id(1)};

    if (GID.x >= numberOfQueues || GID.y >= slistChunkSize)
        return;

    __global uint_float *slist = slistChunks + GID.x * slistChunkSize;
    uint slistSize = slistSizes[GID.x];

    if (GID.y >= slistSize)
        return;

    const uint_float nodeCost = slist[GID.y];
    const Info       nodeInfo = info[nodeCost.first];

    if (nodeInfo.closed == 1)
        continue;

    // TODO: Dedublication with hashing and stuff...

    __global uint_float *tlist = tlistChunks + GID.x * slistChunkSize;
    uint index = atomic_inc(tlistSizes + GID.x);
    tlist[index] = nodeCost;
}

__kernel void computeAndPushBack()
{
}
