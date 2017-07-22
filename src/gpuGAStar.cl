// GPU GA* program

#define GID get_global_id(0)
#define LID get_local_id(0)

// ----- Types ----------------------------------------------------------------
typedef struct {
    uint  first;
    float second;
} uint_float;

// ----- Kernels --------------------------------------------------------------
__kernel void extractAndExpand(const    ulong       numberOfQueues,
                               const    ulong       sizeOfAQueue,
                               __global uint_float *openLists,
                               __global uint       *openSizes
                               __global char       *closed)
{
    if (GID >= numberOfQueues)
        return;

    __global openList = openLists + GID * sizeOfAQueue;
    __global openSize = openSizes + GID;

    if (*openSize == 0)
        return;

    const uint current = top(openList);
    pop(openList, openSize);

    if (current == destination)
        return; // TODO: target found

    closed[current] = 1;

    for (neighbors) {
        // TODO
    }
}

__kernel void checkAndFinalize()
{
    if (GID >= numberOfQueues)
        return;
}

__kernel void duplicateDetection()
{
    if (GID >= numberOfQueues)
        return;
}

__kernel void computeAndPushBack()
{
    if (GID >= numberOfQueues)
        return;
}
