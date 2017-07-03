#pragma once

#include "Graph.h"
#include "Node.h"
#include "Position.h"
#include <vector>

std::vector<Node> cpuAStar(const Graph &g, const Position &source, const Position &destination);
