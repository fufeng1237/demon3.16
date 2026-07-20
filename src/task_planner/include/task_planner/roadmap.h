#ifndef TASK_PLANNER_ROADMAP_H
#define TASK_PLANNER_ROADMAP_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include "task_planner/common.h"

// 定义图的节点类型
const std::string NODE_TYPE_NORMAL = "node";
const std::string NODE_TYPE_PORT = "port";
const std::string NODE_TYPE_TASK_ANCHOR = "task_anchor";
const std::string NODE_TYPE_SHIP = "ship";
const std::string NODE_TYPE_GAS_STATION = "gas_station";

// 定义图的节点
struct GraphNode {
    int id;
    int x;
    int y;
    std::string type; // "node", "port", "ship", "gas_station"
    int source_id = -1;
};

// 定义图的边
struct GraphEdge {
    int to;
    double weight;
};

// 自定义图类 (替代 networkx.Graph)
class Graph {
public:
    std::unordered_map<int, GraphNode> nodes;
    std::unordered_map<int, std::vector<GraphEdge>> adj_list;

    void addNode(int id, int x, int y, const std::string& type, int source_id = -1) {
        nodes[id] = {id, x, y, type, source_id};
    }

    void addEdge(int u, int v, double weight) {
        adj_list[u].push_back({v, weight});
        adj_list[v].push_back({u, weight}); // 无向图
    }

    int getNextNodeId() {
        int max_id = -1;
        for (const auto& pair : nodes) {
            if (pair.first > max_id) max_id = pair.first;
        }
        return max_id + 1;
    }

    // 获取所有加油站节点
    std::vector<GraphNode> getGasStationNodes() {
        std::vector<GraphNode> stations;
        for (const auto& pair : nodes) {
            if (pair.second.type == NODE_TYPE_GAS_STATION) {
                stations.push_back(pair.second);
            }
        }
        return stations;
    }
};

// 声明路网生成函数
// 基础路网生成（港口+USV）
Graph generateRoadmapForAllocation(
    const std::unordered_map<int, Port>& ports,
    const std::vector<USV>& usvs,
    double scale_factor);

// 带加油站集成的路网生成
Graph generateRoadmapWithGasStations(
    const std::unordered_map<int, Port>& ports,
    const std::vector<USV>& usvs,
    const std::vector<GasStation>& gas_stations,
    double scale_factor);

// 加油站连接到最近节点
void connectGasStationToNearestNode(
    Graph& graph,
    const std::vector<GasStation>& gas_stations,
    int n_neighbors = 2);

#endif // TASK_PLANNER_ROADMAP_H
