#pragma once

#include "Graph.h"
#include "Node.h"
#include "Position.h"
#include <vector>

std::vector<Node> cpuAStar(const Graph &graph, const Position &source, const Position &destination);

std::vector<Node> gpuAStar(const Graph &                                     graph,
                           const std::vector<std::pair<Position, Position>> &sourceDestinationList);

std::vector<Node> gpuGAStar(const Graph &graph, const Position &source,
                            const Position &destination);
