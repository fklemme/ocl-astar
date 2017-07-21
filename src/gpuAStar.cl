// GPU A* program

#define DEBUG 0

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

typedef struct {
    uint  closed;
    float totalCost;
    uint  predecessor;
    uint  _reserved;   // for memory alignment
} Info;

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
    for (uint index = 0; index < open->size; ++index) {
        uint_float iValue = _read_heap(open, index);
        if (iValue.first == value)
            return index;
    }

    return open->size;
}

#if DEBUG
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
#endif

// ----- Kernel logic ---------------------------------------------------------
float heuristic(int2 source, int2 destination) {
    const int2 diff = destination - source;
    return sqrt((float) (diff.x * diff.x + diff.y * diff.y));
}

size_t recreate_path(__global const int2  *nodes,
                     __global       int2  *path,
                                    ulong  maxPathLength,
                     __global       Info  *info,
                                    uint   destination)
{
    // TODO: optimize! Use local memory!

    path[0] = nodes[destination];
    size_t length = 1;

    uint node        = destination;
    uint predecessor = info[node].predecessor;

    while (length < maxPathLength && node != predecessor) {
        node           = predecessor;
        predecessor    = info[node].predecessor;
        path[length++] = nodes[node];
    }

    // reverse path
    for (int i = 0; i < length / 2; ++i) {
        int2 tmp = path[i];
        path[i] = path[length - i - 1];
        path[length - i - 1] = tmp;
    }

    return length;
}

__kernel void gpuAStar(__global const int2       *nodes,            // x, y
                                const ulong       nodesSize,
                       __global const uint_float *edges,            // destination index, stepCost
                                const ulong       edgesSize,
                       __global const uint2      *adjacencyMap,     // edges_begin, edges_end
                                const ulong       adjacencyMapSize,
                                const ulong       numberOfAgents,   // provides offset for per-thread arguments below
         /* input:  */ __global const uint2      *srcDstList,       // source id, destination id
         /* output: */ __global       int2       *paths,            // x, y; offset = GID * maxPathLength;
                                const ulong       maxPathLength,
                       __local        uint_float *openLocal,        // open lists: id, cost
                                const ulong       openLocalSize,    // per agent (local memory) open list size
                       __global       uint_float *openGlobalExt,    // open lists: id, cost; fallback if out of local memory
                       __global       Info       *infos,            // closed lists, see members at the top
                       __global       int        *returnCode)       // return code (for debugging purposes)
{
    const size_t GID = get_global_id(0);
    const size_t LID = get_local_id(0);

    if (GID >= numberOfAgents)
        return;

    OpenList open = {openLocal + LID * openLocalSize,
                     openLocalSize,
                     openGlobalExt + GID * nodesSize,
                     0};

    __global Info *info = infos + GID * nodesSize;

    const uint source      = srcDstList[GID].x;
    const uint destination = srcDstList[GID].y;

    // Initialize result in case no path is found.
    // If the first node in path is not source, we can expect a failure.
    paths[GID * maxPathLength] = nodes[destination];
    returnCode[GID] = 1; // failure: no path found!

    info[source].predecessor = source; // to recreate path

    // Begin at source
    push(&open, source, 0.0f);

    while (open.size > 0) {
        const uint current = top(&open);
        pop(&open);

#if DEBUG
        // DEBUG: heap after pop
        if (!is_heap(&open)) {
            returnCode[GID] = 90;
            return; // error: broken heap!
        }
#endif

        if (current == destination) {
            size_t length = recreate_path(nodes, paths + GID * maxPathLength,
                                          maxPathLength, info, destination);
            returnCode[GID] = length < maxPathLength ?
                0 : // success: path found!
                2;  // failure: path too long!
            return;
        }

        info[current].closed = 1; // close node
        const float totalCost = info[current].totalCost;

        const uint2 edgeRange = adjacencyMap[current];
        for (uint edge = edgeRange.x; edge != edgeRange.y; ++edge) {
            const uint  nbNode     = edges[edge].first;
            const float nbStepCost = edges[edge].second;

            if (info[nbNode].closed == 1)
                continue;

            const float nbTotalCost = totalCost + nbStepCost;
            const uint  nbIndex = find(&open, nbNode);

            if (nbIndex < open.size && info[nbNode].totalCost <= nbTotalCost)
                continue;

            info[nbNode].totalCost = nbTotalCost;

            // Store predecessor to recreate path
            info[nbNode].predecessor = current;

            const float nbHeuristic = heuristic(nodes[nbNode], nodes[destination]);

            if (nbIndex < open.size)
                update(&open, nbIndex, nbNode, nbTotalCost + nbHeuristic);
            else
                push(&open, nbNode, nbTotalCost + nbHeuristic);

#if DEBUG
            // DEBUG: heap after push / update
            if (!is_heap(&open)) {
                returnCode[GID] = 91;
                return; // error: broken heap!
            }
#endif
        }
    }
}
