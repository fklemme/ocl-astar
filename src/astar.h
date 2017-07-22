#pragma once

#include "Graph.h"
#include "Node.h"
#include "Position.h"
#include <vector>

// Enable printing of debug information from functions below.
#define DEBUG_OUTPUT

std::vector<Node> cpuAStar(const Graph &graph, const Position &source, const Position &destination);

std::vector<std::vector<Node>>
gpuAStar(const Graph &graph, const std::vector<std::pair<Position, Position>> &srcDstList);

std::vector<Node> gpuGAStar(const Graph &graph, const Position &source,
                            const Position &destination);
