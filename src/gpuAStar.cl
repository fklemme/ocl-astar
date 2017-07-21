// GPU A* program

// ----- Types ----------------------------------------------------------------
typedef struct {
    uint  first;
    float second;
} uint_float;

typedef struct {
    __local  uint_float *localMem;
    const    size_t      localSize;
    __global uint_float *globalExt;
             size_t      size;
} OpenList;

// ----- Helper ---------------------------------------------------------------
uint_float _read_heap(OpenList *open, size_t index) {
    return index < open->localSize ?
        open->localMem[index] :
        open->globalExt[index - open->localSize];
}

void _write_heap(OpenList *open, size_t index, uint_float value) {
    if (index < open->localSize)
        open->localMem[index] = value;
    else
        open->globalExt[index - open->localSize] = value;
}

// ----- OpenList functions ---------------------------------------------------
uint top(OpenList *open) {
    return open->localMem[0].first;
}

void _push_impl(OpenList *open, size_t *size, uint value, float cost) {
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

void push(OpenList *open, uint value, float cost) {
    _push_impl(open, &open->size, value, cost);
}

void update(OpenList *open, size_t index, uint value, float cost) {
    _push_impl(open, &index, value, cost);
}

void pop(OpenList *open) {
    uint_float value = _read_heap(open, --(open->size));
    size_t     index = 0;

    while (index < open->size / 2) {
        size_t child = index * 2 + 1;

        uint_float cValue = _read_heap(open, child);
        if (child + 1 < open->size) {
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

uint find(OpenList *open, uint value) {
    uint index = 0;

    while (index < open->size) {
        uint_float iValue = _read_heap(open, index);
        if (iValue.first == value)
            break;
    }

    return index;
}

// Debugging / testing
bool is_heap(OpenList *open) {
    for (size_t index = 0; index < open->size / 2; ++index) {
        uint_float value = _read_heap(open, index);
        size_t     left   = index * 2 + 1;
        size_t     right  = index * 2 + 2;

        uint_float lValue = _read_heap(open, left);
        if (lValue.second < value.second)
            return false;

        if (right < open->size) {
            uint_float rValue = _read_heap(open, right);
            if (rValue.second < value.second)
                return false;
        }
    }

    return true;
}

// ----- Kernel ---------------------------------------------------------------
float heuristic(int2 source, int2 destination) {
    const int2 diff = destination - source;
    return sqrt((float) (diff.x * diff.x + diff.y * diff.y));
}

__kernel void gpuAStar(__global const int2       *nodes,            // x, y
                                const ulong       nodesSize,
                       __global const uint_float *edges,            // destination index, cost
                                const ulong       edgesSize,
                       __global const uint2      *adjacencyMap,     // edges_begin, edges_end
                                const ulong       adjacencyMapSize,
                                const ulong       numberOfAgents,   // provides offset for per-thread arguments below
         /* input:  */ __global const uint2      *srcDstList,       // source id, destination id
         /* output: */ __global       int2       *paths,            // x, y; offset = GID * maxPathLength;
                                const ulong       maxPathLength,
                       __local        uint_float *openLocal,        // id, cost
                                const ulong       openLocalSize,    // per agent local memory
                       __global       uint_float *openGlobalExt,    // id, cost; fallback if out of local memory
                       __global       float      *info)             // total cost; or -1.0f ^= visited
{
    const size_t GID = get_global_id(0);
    const size_t LID = get_local_id(0);

    if (GID >= numberOfAgents)
        return;

    OpenList open = {openLocal + LID * openLocalSize,
                     openLocalSize,
                     openGlobalExt + GID * nodesSize,
                     0};

    const uint source      = srcDstList[GID].x;
    const uint destination = srcDstList[GID].y;

    // Initialize result in case no path is found.
    // If the first node in path is not source, we can expect a failure.
    paths[GID * maxPathLength] = nodes[destination];

    // Begin at source
    push(&open, source, 0.0f);

    while (open.size > 0) {
        const uint current = top(&open);
        pop(&open);

        if (current == destination)
            break; // TODO

        const float totalCost = info[current];
        info[current] = -1.0f; // close

        const uint2 neighbors = adjacencyMap[current];
        for (uint neighbor = neighbors.x; neighbor != neighbors.y; ++neighbor) {
            const uint  nbNode     = edges[neighbor].first;
            const float nbStepCost = edges[neighbor].second;

            if (info[current] < 0.0f) // closed
                continue;

            const float nbTotalCost = totalCost + nbStepCost;
            const uint  nbIndex = find(&open, neighbor);

            if (nbIndex < open.size && info[neighbor] <= nbTotalCost)
                continue;

            const float nbHeuristic = heuristic(nodes[neighbor], nodes[destination]);

            if (nbIndex < open.size)
                update(&open, nbIndex, nbNode, nbTotalCost + nbHeuristic);
            else
                push(&open, nbNode, nbTotalCost + nbHeuristic);
        }
    }
}
