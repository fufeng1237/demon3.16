/**
 * Standalone road network generator — uses usv_roadmap_standalone code.
 * Replaces ROS logging with printf, reads configs, outputs graph as JSON.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <regex>

// Replace ROS logging
#define ROS_INFO(...) printf(__VA_ARGS__); printf("\n")
#define ROS_WARN(...) fprintf(stderr, "WARN: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#define ROS_ERROR(...) fprintf(stderr, "ERROR: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#define ROS_DEBUG(...)

// Include the roadmap generation code
#include "task_planner/roadmap.h"
#include "task_planner/common.h"

// Override the map path
static std::string g_map_file = "/root/demon3.16/data/maps/binary_map_scaled.png";

// Since generateRoadmapForAllocation hardcodes the map path, we need to modify it.
// We'll make a copy with our map path.

// Actually, let's just include and modify the source directly.
// The simplest approach: redefine the map path as a global that the function uses.

// Hmm, the function hardcodes the path. Let me just copy the function body
// into a new function that takes the map path as parameter.

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <queue>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================
// Copy of roadmap.cpp functions with configurable map path
// ============================================================

struct PointCmp {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    }
};

struct EdgeCmp {
    bool operator()(const std::pair<cv::Point, cv::Point>& a, const std::pair<cv::Point, cv::Point>& b) const {
        PointCmp pcmp;
        if (pcmp(a.first, b.first)) return true;
        if (pcmp(b.first, a.first)) return false;
        return pcmp(a.second, b.second);
    }
};

double calcDistance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

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

std::pair<cv::Point, cv::Point> make_edge(const cv::Point& p1, const cv::Point& p2) {
    PointCmp cmp;
    if (cmp(p1, p2)) return {p1, p2};
    return {p2, p1};
}

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

Graph generateRoadmapForAllocation(const std::string& map_file,
    const std::unordered_map<int, Port>& ports,
    const std::vector<USV>& usvs, double scale_factor) {

    Graph graph;
    cv::Mat map_img = cv::imread(map_file, cv::IMREAD_GRAYSCALE);
    if (map_img.empty()) {
        printf("ERROR: Cannot read map: %s\n", map_file.c_str());
        return graph;
    }
    printf("Map loaded: %dx%d\n", map_img.cols, map_img.rows);

    // 1. Skeleton
    cv::Mat binary, skeleton;
    cv::threshold(map_img, binary, 127, 255, cv::THRESH_BINARY);
    cv::ximgproc::thinning(binary, skeleton, cv::ximgproc::THINNING_GUOHALL);
    printf("Skeleton: %d pixels\n", cv::countNonZero(skeleton));

    // 2. Critical pixels
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
    printf("Critical pixels: %zu\n", critical_pixels.size());

    // 3. DBSCAN clustering (eps=25)
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
    printf("Clusters: %d\n", cluster_id);

    // 4. Path tracing
    std::set<std::pair<cv::Point, cv::Point>, EdgeCmp> visited_edges;
    struct EdgeData { double weight; };
    std::map<std::pair<int, int>, EdgeData> final_edges;

    for (const auto& start_node : critical_pixels) {
        auto neighbors = getSkeletonNeighbors(skeleton, start_node);
        for (const auto& neighbor : neighbors) {
            auto edge = make_edge(start_node, neighbor);
            if (visited_edges.count(edge)) continue;
            visited_edges.insert(edge);
            cv::Point prev = start_node, curr = neighbor;
            double path_length = calcDistance(prev, curr);
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
                prev = curr; curr = next_node;
            }
            cv::Point end_node = curr;
            int c1_id = raw_to_cluster_id[start_node];
            int c2_id = raw_to_cluster_id[end_node];
            if (c1_id != c2_id) {
                auto cluster_edge = std::make_pair(std::min(c1_id, c2_id), std::max(c1_id, c2_id));
                double real_length = path_length / scale_factor;
                if (final_edges.find(cluster_edge) == final_edges.end() ||
                    real_length < final_edges[cluster_edge].weight) {
                    final_edges[cluster_edge] = {real_length};
                }
            }
        }
    }
    printf("Cluster edges: %zu\n", final_edges.size());

    // 5. Build graph
    std::map<int, int> cluster_to_graph_id;
    for (int i = 0; i < cluster_id; ++i) {
        int node_id = graph.getNextNodeId();
        graph.addNode(node_id, cluster_centroids[i].x / scale_factor,
                      cluster_centroids[i].y / scale_factor, "node");
        cluster_to_graph_id[i] = node_id;
    }
    for (const auto& kv : final_edges) {
        int u = cluster_to_graph_id[kv.first.first];
        int v = cluster_to_graph_id[kv.first.second];
        graph.addEdge(u, v, kv.second.weight);
    }
    printf("Base graph: %zu nodes, %zu adjacency entries\n", graph.nodes.size(), graph.adj_list.size());

    // 6. Inject ports and ships
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
    printf("After anchors: %zu nodes\n", graph.nodes.size());

    // 7. Phase 1 prune (isolated)
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> to_remove;
        for (const auto& pair : graph.nodes) {
            if (pair.second.type != "node") continue;
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

    // 8. Phase 2 prune (dead-end)
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
            if (graph.nodes[neighbor_id].type == "node") {
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
    printf("After prune: %zu nodes\n", graph.nodes.size());
    return graph;
}

void connectGasStationToNearestNode(Graph& graph,
    const std::vector<GasStation>& gas_stations, int n_neighbors) {
    for (const auto& station : gas_stations) {
        int nearest_node = -1;
        double min_dist = std::numeric_limits<double>::infinity();
        cv::Point station_pos(station.x, station.y);
        for (const auto& pair : graph.nodes) {
            if (pair.second.type == "gas_station") continue;
            double dist = calcDistance(station_pos, cv::Point(pair.second.x, pair.second.y));
            if (dist < min_dist) { min_dist = dist; nearest_node = pair.first; }
        }
        if (nearest_node != -1) {
            int sid = graph.getNextNodeId();
            graph.addNode(sid, station.x, station.y, "gas_station");
            graph.addEdge(sid, nearest_node, min_dist);
            printf("Gas station %d -> node %d (dist=%.1f)\n", station.id, sid, min_dist);
            if (n_neighbors > 1) {
                std::vector<std::pair<double, int>> distances;
                for (const auto& pair : graph.nodes) {
                    if (pair.first == sid) continue;
                    if (pair.second.type == "gas_station") continue;
                    double dist = calcDistance(station_pos, cv::Point(pair.second.x, pair.second.y));
                    distances.push_back({dist, pair.first});
                }
                std::sort(distances.begin(), distances.end());
                for (int i = 1; i < std::min(n_neighbors, (int)distances.size()); ++i) {
                    graph.addEdge(sid, distances[i].second, distances[i].first);
                }
            }
        }
    }
}

// ============================================================
// JSON Output
// ============================================================

void outputGraphJSON(const Graph& graph, const std::string& output_path) {
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"n_nodes\": " << graph.nodes.size() << ",\n";
    out << "  \"n_edges\": 0,\n";  // Count below
    out << "  \"nodes\": [\n";
    bool first = true;
    for (const auto& pair : graph.nodes) {
        if (!first) out << ",\n";
        first = false;
        const auto& n = pair.second;
        bool is_port = (n.type == "port");
        bool is_gas = (n.type == "gas_station");
        out << "    {\"id\": " << n.id << ", \"x\": " << n.x << ", \"y\": " << n.y
            << ", \"is_port\": " << (is_port ? "true" : "false")
            << ", \"is_gas_station\": " << (is_gas ? "true" : "false")
            << ", \"port_name\": \"" << (is_port ? "Port_" + std::to_string(n.id) : (is_gas ? "Gas_" + std::to_string(n.id) : "")) << "\""
            << ", \"type\": \"" << n.type << "\""
            << ", \"degree\": 0}";
    }
    out << "\n  ],\n";

    // Count edges
    int edge_count = 0;
    std::set<std::pair<int,int>> seen;
    out << "  \"edges\": [\n";
    first = true;
    for (const auto& pair : graph.adj_list) {
        for (const auto& edge : pair.second) {
            int u = pair.first, v = edge.to;
            if (u > v) std::swap(u, v);
            if (seen.count({u, v})) continue;
            seen.insert({u, v});
            if (!first) out << ",\n";
            first = false;
            out << "    {\"from\": " << u << ", \"to\": " << v << ", \"weight\": " << edge.weight << "}";
            edge_count++;
        }
    }
    out << "\n  ],\n";
    out << "  \"n_edges\": " << edge_count << ",\n";
    out << "  \"distance_matrix\": null\n";
    out << "}\n";
    out.close();
    printf("Output: %s (%zu nodes, %d edges)\n", output_path.c_str(), graph.nodes.size(), edge_count);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    std::string map_file = "/root/demon3.16/data/maps/binary_map_scaled.png";
    std::string ports_file = "/root/demon3.16/src/task_planner/cpp_standalone/ports.txt";
    std::string usvs_file = "/root/demon3.16/src/task_planner/cpp_standalone/usvs.txt";
    std::string gas_file = "/root/demon3.16/src/task_planner/cpp_standalone/gas_stations.txt";
    std::string output_file = "/root/demon3.16/src/task_planner/output/road_network_cpp.json";
    double scale_factor = 1.0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--map" && i+1 < argc) map_file = argv[++i];
        else if (arg == "--ports" && i+1 < argc) ports_file = argv[++i];
        else if (arg == "--usvs" && i+1 < argc) usvs_file = argv[++i];
        else if (arg == "--gas" && i+1 < argc) gas_file = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_file = argv[++i];
        else if (arg == "--scale" && i+1 < argc) scale_factor = std::stod(argv[++i]);
    }

    printf("=== Road Network Generator (C++ standalone) ===\n");
    printf("Map: %s\n", map_file.c_str());

    auto ports = parsePortFile(ports_file);
    auto usvs = parseUSVFile(usvs_file);
    auto gas_stations = parseGasStationFile(gas_file);
    printf("Loaded: %zu ports, %zu USVs, %zu gas stations\n", ports.size(), usvs.size(), gas_stations.size());

    Graph graph = generateRoadmapForAllocation(map_file, ports, usvs, scale_factor);
    if (!gas_stations.empty()) {
        connectGasStationToNearestNode(graph, gas_stations, 2);
    }

    printf("Final: %zu nodes\n", graph.nodes.size());
    outputGraphJSON(graph, output_file);
    return 0;
}
