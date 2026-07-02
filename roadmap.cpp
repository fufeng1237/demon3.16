#include "usv_roadmap/roadmap.h"
#include <ros/ros.h>
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> // 必须包含此头文件以使用细化算法

// 辅助：二维点比较器（用于 std::map）
struct PointCmp {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    }
};
// 辅助：边的比较器（用于 std::set）
struct EdgeCmp {
    bool operator()(const std::pair<cv::Point, cv::Point>& a, const std::pair<cv::Point, cv::Point>& b) const {
        PointCmp pcmp;
        if (pcmp(a.first, b.first)) return true;
        if (pcmp(b.first, a.first)) return false;
        return pcmp(a.second, b.second);
    }
};

// 计算欧氏距离
double calcDistance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// 辅助：获取骨架图像中某个点的 8 邻域骨架点
std::vector<cv::Point> getSkeletonNeighbors(const cv::Mat& skel, const cv::Point& p) {
    std::vector<cv::Point> neighbors;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = p.x + dx, ny = p.y + dy;
            if (nx >= 0 && nx < skel.cols && ny >= 0 && ny < skel.rows) {
                if (skel.at<uchar>(ny, nx) > 127) {
                    neighbors.push_back(cv::Point(nx, ny));
                }
            }
        }
    }
    return neighbors;
}

// 规范化边（保证小的点在前面，方便作为 set 的键）
std::pair<cv::Point, cv::Point> make_edge(const cv::Point& p1, const cv::Point& p2) {
    PointCmp cmp;
    if (cmp(p1, p2)) return {p1, p2};
    return {p2, p1};
}

// 完全一比一复刻 Python 的 connect_to_nearest 逻辑
void connectToNearestPythonStyle(Graph& G, int temp_node_id, int n_neighbors, double scale) {
    GraphNode target = G.nodes[temp_node_id];
    cv::Point target_pos(target.x, target.y);
    std::vector<std::pair<double, int>> distances;

    for (const auto& pair : G.nodes) {
        if (pair.first == temp_node_id) continue;
        GraphNode node = pair.second;
        
        bool allow_connect = false;
        if (node.type == "node") {
            allow_connect = true;
        } else if (target.type == "ship" && node.type == "port") {
            allow_connect = true;
        } else if (target.type == "port" && node.type == "ship") {
            allow_connect = true;
        }

        if (allow_connect) {
            double dist = calcDistance(target_pos, cv::Point(node.x, node.y));
            distances.push_back({dist, pair.first});
        }
    }

    std::sort(distances.begin(), distances.end());

    if (distances.empty()) return;

    if (target.type == "port") {
        for (int i = 0; i < std::min(n_neighbors, (int)distances.size()); ++i) {
            G.addEdge(temp_node_id, distances[i].second, distances[i].first);
        }
    } else if (target.type == "ship") {
        G.addEdge(temp_node_id, distances[0].second, distances[0].first);
        if (n_neighbors >= 2 && distances.size() >= 2) {
            double diff = distances[1].first / (distances[0].first + 1e-6);
            if (diff < 1.5) {
                G.addEdge(temp_node_id, distances[1].second, distances[1].first);
            }
        }
    }
}

Graph generateRoadmapForAllocation(const std::unordered_map<int, Port>& ports, const std::vector<USV>& usvs, double scale_factor) {
    Graph graph;
    std::string scaled_map_file = "/demon04.07/data/maps/binary_map_scaled_small.png";
    cv::Mat map_img = cv::imread(scaled_map_file, cv::IMREAD_GRAYSCALE);
    if (map_img.empty()) {
        ROS_ERROR("无法读取缩放地图!");
        return graph;
    }

    // 1. 骨架提取 (等价于 Python 的 skeletonize)
    cv::Mat binary, skeleton;
    cv::threshold(map_img, binary, 127, 255, cv::THRESH_BINARY);
    cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_GUOHALL);

    // 2. 识别关键节点 (度数 != 2 的点：端点和交叉点)
    std::vector<cv::Point> critical_pixels;
    std::map<cv::Point, int, PointCmp> pixel_degrees;
    
    for (int y = 0; y < skeleton.rows; ++y) {
        for (int x = 0; x < skeleton.cols; ++x) {
            if (skeleton.at<uchar>(y, x) > 127) {
                cv::Point p(x, y);
                auto neighbors = getSkeletonNeighbors(skeleton, p);
                pixel_degrees[p] = neighbors.size();
                if (neighbors.size() != 2) {
                    critical_pixels.push_back(p);
                }
            }
        }
    }

    if (critical_pixels.size() < 2) {
        ROS_WARN("关键节点不足，无法形成网络！");
        return graph;
    }

    // 3. DBSCAN 节点聚类 (增大 eps 值以减少节点数量)
    // eps=30 表示距离小于30像素的关键像素会被聚为一类
    std::vector<int> labels(critical_pixels.size(), -1);
    int cluster_id = 0;
    for (size_t i = 0; i < critical_pixels.size(); ++i) {
        if (labels[i] != -1) continue;
        std::queue<int> q;
        q.push(i);
        labels[i] = cluster_id;
        while (!q.empty()) {
            int curr = q.front(); q.pop();
            for (size_t j = 0; j < critical_pixels.size(); ++j) {
                if (labels[j] == -1 && calcDistance(critical_pixels[curr], critical_pixels[j]) <= 25.0) {
                    labels[j] = cluster_id;
                    q.push(j);
                }
            }
        }
        cluster_id++;
    }

    // 计算每个聚类的中心点，并建立映射
    std::vector<cv::Point> cluster_centroids(cluster_id);
    std::vector<int> cluster_counts(cluster_id, 0);
    std::map<cv::Point, int, PointCmp> raw_to_cluster_id;

    for (size_t i = 0; i < critical_pixels.size(); ++i) {
        int cid = labels[i];
        cluster_centroids[cid].x += critical_pixels[i].x;
        cluster_centroids[cid].y += critical_pixels[i].y;
        cluster_counts[cid]++;
        raw_to_cluster_id[critical_pixels[i]] = cid;
    }

    for (int i = 0; i < cluster_id; ++i) {
        cluster_centroids[i].x /= cluster_counts[i];
        cluster_centroids[i].y /= cluster_counts[i];
    }

    // 4. 追踪路径 (Trace Paths) - 用来连线
    std::set<std::pair<cv::Point, cv::Point>, EdgeCmp> visited_edges;
    struct EdgeData { double weight; };
    std::map<std::pair<int, int>, EdgeData> final_edges; // cluster_id1-cluster_id2 -> EdgeData

    for (const auto& start_node : critical_pixels) {
        auto neighbors = getSkeletonNeighbors(skeleton, start_node);
        for (const auto& neighbor : neighbors) {
            auto edge = make_edge(start_node, neighbor);
            if (visited_edges.count(edge)) continue;

            visited_edges.insert(edge);
            cv::Point prev = start_node;
            cv::Point curr = neighbor;
            double path_length = calcDistance(prev, curr);

            // 一直走到下一个关键节点
            while (pixel_degrees[curr] == 2) {
                auto next_neighbors = getSkeletonNeighbors(skeleton, curr);
                cv::Point next_node(-1, -1);
                for (const auto& nn : next_neighbors) {
                    if (nn != prev) { next_node = nn; break; }
                }
                if (next_node.x == -1) break;

                auto next_edge = make_edge(curr, next_node);
                visited_edges.insert(next_edge);
                path_length += calcDistance(curr, next_node);

                prev = curr;
                curr = next_node;
            }

            cv::Point end_node = curr;
            int c1_id = raw_to_cluster_id[start_node];
            int c2_id = raw_to_cluster_id[end_node];

            if (c1_id != c2_id) {
                auto cluster_edge = std::make_pair(std::min(c1_id, c2_id), std::max(c1_id, c2_id));
                double real_length = path_length / scale_factor;
                
                // 保留最短路径
                if (final_edges.find(cluster_edge) == final_edges.end() ||
                    real_length < final_edges[cluster_edge].weight) {
                    final_edges[cluster_edge] = {real_length};
                }
            }
        }
    }

    // 5. 将聚类中心写入底层图，并直接连线（不添加中间节点）
    std::map<int, int> cluster_to_graph_id; // cluster_id -> graph_node_id
    for (int i = 0; i < cluster_id; ++i) {
        int node_id = graph.getNextNodeId();
        // 缩放回原始尺寸
        graph.addNode(node_id, cluster_centroids[i].x / scale_factor, cluster_centroids[i].y / scale_factor, "node");
        cluster_to_graph_id[i] = node_id;
    }

    // 直接连接两个拐点，不添加中间节点
    for (const auto& kv : final_edges) {
        int u = cluster_to_graph_id[kv.first.first];
        int v = cluster_to_graph_id[kv.first.second];
        graph.addEdge(u, v, kv.second.weight);
    }

    // 6. 先接入港口、船只、任务起点终点 (作为锚点，防止后续剪枝断开路网)
    ROS_INFO("接入港口和船只作为锚点...");
    for (const auto& pair : ports) {
        int port_id = graph.getNextNodeId();
        graph.addNode(port_id, pair.second.x, pair.second.y, "port");
        connectToNearestPythonStyle(graph, port_id, 1, scale_factor);
    }
    for (const auto& usv : usvs) {
        int ship_id = graph.getNextNodeId();
        graph.addNode(ship_id, usv.x, usv.y, "ship");
        connectToNearestPythonStyle(graph, ship_id, 2, scale_factor);
    }
    ROS_INFO("锚点接入完成！当前节点: %zu", graph.nodes.size());

    // 7. 剪枝 - 只移除度数为0的孤立普通节点 (保留与港口/船舶相连的骨架)
    ROS_INFO("开始剪枝 (保留锚点连接)...");
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> to_remove;
        for (const auto& pair : graph.nodes) {
            if (pair.second.type != "node") continue;  // 只处理普通节点
            // 只移除度数为0的完全孤立节点
            if (graph.adj_list[pair.first].size() == 0) {
                to_remove.push_back(pair.first);
                changed = true;
            }
        }
        for (int id : to_remove) {
            graph.adj_list.erase(id);
            graph.nodes.erase(id);
        }
    }

    // 8. 温和剪枝：移除度数为1的普通节点，但仅当其邻居也是普通节点时
    //     (如果邻居是 port/ship/gas_station，则保留该骨架分支)
    ROS_INFO("温和剪枝：移除无锚点死胡同...");
    changed = true;
    while (changed) {
        changed = false;
        std::vector<int> to_remove;
        for (const auto& pair : graph.nodes) {
            int node_id = pair.first;
            const GraphNode& node = pair.second;
            if (node.type != "node") continue;
            if (graph.adj_list[node_id].size() != 1) continue;

            int neighbor_id = graph.adj_list[node_id][0].to;
            if (graph.nodes.find(neighbor_id) == graph.nodes.end()) continue;
            const GraphNode& neighbor = graph.nodes[neighbor_id];
            // 只有当邻居也是普通节点(非锚点)时才移除
            if (neighbor.type == "node") {
                to_remove.push_back(node_id);
                changed = true;
            }
        }
        for (int id : to_remove) {
            for (auto& edge : graph.adj_list[id]) {
                int neighbor = edge.to;
                auto& neighbor_edges = graph.adj_list[neighbor];
                for (auto it = neighbor_edges.begin(); it != neighbor_edges.end(); ) {
                    if (it->to == id) { it = neighbor_edges.erase(it); }
                    else { ++it; }
                }
            }
            graph.adj_list.erase(id);
            graph.nodes.erase(id);
        }
    }

    ROS_INFO("路网拓扑提取完成！保留节点: %zu", graph.nodes.size());
    return graph;
    }
    
    // ============================================================
    // 加油站连接到最近节点 [新增]
    // ============================================================
    void connectGasStationToNearestNode(Graph& graph,
                                          const std::vector<GasStation>& gas_stations,
                                          int n_neighbors) {
        for (const auto& station : gas_stations) {
            // 找到最近的节点
            int nearest_node = -1;
            double min_dist = std::numeric_limits<double>::infinity();
            cv::Point station_pos(station.x, station.y);
            
            for (const auto& pair : graph.nodes) {
                if (pair.second.type == NODE_TYPE_GAS_STATION) continue;  // 跳过其他加油站
                
                double dist = calcDistance(station_pos, cv::Point(pair.second.x, pair.second.y));
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_node = pair.first;
                }
            }
            
            if (nearest_node != -1) {
                // 添加加油站节点
                int station_node_id = graph.getNextNodeId();
                graph.addNode(station_node_id, station.x, station.y, NODE_TYPE_GAS_STATION);
                
                // 连接到最近的节点
                graph.addEdge(station_node_id, nearest_node, min_dist);
                ROS_INFO("加油站 %d 连接到节点 %d，距离: %.2f",
                         station.id, nearest_node, min_dist);
                
                // 如果有多个邻居配置，尝试连接更多近邻
                if (n_neighbors > 1) {
                    std::vector<std::pair<double, int>> distances;
                    for (const auto& pair : graph.nodes) {
                        if (pair.first == station_node_id) continue;
                        if (pair.second.type == NODE_TYPE_GAS_STATION) continue;
                        
                        double dist = calcDistance(station_pos, cv::Point(pair.second.x, pair.second.y));
                        distances.push_back({dist, pair.first});
                    }
                    std::sort(distances.begin(), distances.end());
                    
                    for (int i = 1; i < std::min(n_neighbors, (int)distances.size()); ++i) {
                        graph.addEdge(station_node_id, distances[i].second, distances[i].first);
                        ROS_INFO("加油站 %d 额外连接到节点 %d，距离: %.2f",
                                 station.id, distances[i].second, distances[i].first);
                    }
                }
            }
        }
    }
    
    // ============================================================
    // 带加油站集成的路网生成 [新增]
    // ============================================================
    Graph generateRoadmapWithGasStations(
        const std::unordered_map<int, Port>& ports,
        const std::vector<USV>& usvs,
        const std::vector<GasStation>& gas_stations,
        double scale_factor) {
        
        // 1. 生成基础路网
        Graph graph = generateRoadmapForAllocation(ports, usvs, scale_factor);
        
        // 2. 集成加油站
        if (!gas_stations.empty()) {
            ROS_INFO("集成 %zu 个加油站到路网...", gas_stations.size());
            connectGasStationToNearestNode(graph, gas_stations, 2);
        }
        
        ROS_INFO("带加油站的路网生成完成！总节点数: %zu", graph.nodes.size());
        return graph;
    }