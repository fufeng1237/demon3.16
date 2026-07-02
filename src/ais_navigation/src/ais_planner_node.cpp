/**
 * ais_planner_node.cpp
 * A* path planner with AIS navigation potential field guidance.
 *
 * Method (ref: 基于AIS导航势场的无人艇路径规划.md §四):
 *   Lane compliance cost:  c_lane(p; d) = max(0, -d * Phi_bar(p))
 *   Step cost:             c(p_i, p_j; d) = ||p_j - p_i|| + lambda * c_lane(p_j; d)
 *
 * Where:
 *   Phi_bar(p) in [-1, 1]: normalized navigation potential field
 *     +1 = downstream channel, -1 = upstream channel, 0 = mixed/free water
 *   d in {+1, -1}: desired direction (+1=downstream, -1=upstream)
 *   lambda > 0: lane compliance weight
 *
 * Publishes:
 *   /ais_planned_path      - lane-guided A* path
 *   /ais_traditional_path  - traditional A* path (pure distance)
 */

#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using namespace grid_map;

// ====================== A* Planner ======================

struct Node {
    int x, y;
    double g, h;
    Node* parent;
    Node(int x_, int y_, double g_, double h_, Node* p = nullptr)
        : x(x_), y(y_), g(g_), h(h_), parent(p) {}
    double f() const { return g + h; }
};

struct NodeCmp {
    bool operator()(Node* a, Node* b) { return a->f() > b->f(); }
};

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

class AISPlanner {
public:
    AISPlanner() : use_local_dir_(true), lambda_(200.0), lambda_qual_(1.0) {}

    void setUseLocalDir(bool v) { use_local_dir_ = v; }
    void setLambda(double lam) { lambda_ = lam; }
    void setLambdaQual(double lam_q) { lambda_qual_ = lam_q; }
    double getLambda() const { return lambda_; }

    // ---- Bounds and obstacle checks ----

    bool isInsideSafe(const GridMap& map, const Index& index) {
        return (index(0) >= 0 && index(0) < map.getSize()(0) &&
                index(1) >= 0 && index(1) < map.getSize()(1));
    }

    bool isBlocked(const GridMap& map, const Index& index) {
        if (!isInsideSafe(map, index)) return true;
        if (map.exists("obstacle")) {
            float val = map.at("obstacle", index);
            if (val > 80.0) return true;
        }
        return false;
    }

    Index findValidNear(const GridMap& map, const Index& index) {
        if (!isBlocked(map, index)) return index;
        for (int r = 1; r <= 10; r++) {
            for (int dx = -r; dx <= r; dx++) {
                for (int dy = -r; dy <= r; dy++) {
                    Index n = index + Index(dx, dy);
                    if (isInsideSafe(map, n) && !isBlocked(map, n)) return n;
                }
            }
        }
        return index;
    }

    // ---- Lane Compliance Cost (direction penalty + position quality) ----

    double laneComplianceCost(const GridMap& map, const Index& cur,
                              const Index& next) const {
        if (!map.exists("phibar")) return 0.0;
        float phi = map.at("phibar", next);
        double signal = std::abs(static_cast<double>(phi));

        if (!use_local_dir_ || !map.exists("dir_x") || !map.exists("dir_y")) {
            // Fallback: legacy scalar d mode
            if (signal < 0.01) return 0.0;
            return lambda_ * std::max(0.0,
                -direction_ * static_cast<double>(phi));
        }

        // Step vector
        Position p_cur, p_next;
        map.getPosition(cur, p_cur);
        map.getPosition(next, p_next);
        double dx = p_next.x() - p_cur.x();
        double dy = p_next.y() - p_cur.y();
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 1e-6) return 0.0;

        // Expected direction at target cell
        float edx = map.at("dir_x", next);
        float edy = map.at("dir_y", next);
        double emag = std::sqrt(edx * edx + edy * edy);
        if (emag < 0.01) return 0.0;  // no direction data → free

        double v_dot_d = (dx * edx + dy * edy) / (dist * emag);

        // Term 1: Direction penalty (strong, only when going wrong way)
        // max(0, -(v·d)): +1 opposite, 0 aligned or perpendicular
        double dir_penalty = lambda_ * std::max(0.0, -v_dot_d) * signal * signal;

        // Term 2: Position quality (weak, proportional to step length)
        // (1 - |Φ̄|): 0 at center → cost=0, 1 at edge → cost≈step
        double qual_penalty = lambda_qual_ * (1.0 - signal) * dist;

        return dir_penalty + qual_penalty;
    }

    // ---- Step Cost ----

    double stepCost(const GridMap& map, const Index& cur, const Index& next,
                    bool use_guidance) const {
        double dist = (next - cur).matrix().cast<double>().norm()
                      * map.getResolution();

        if (!use_guidance) {
            return dist;  // Traditional A*: pure Euclidean distance
        }

        // Lane-guided A*: distance + lane compliance penalty
        return dist + laneComplianceCost(map, cur, next);
    }

    // ---- A* Search ----

    std::vector<Index> plan(const GridMap& map, Index start, Index goal,
                            bool use_guidance) {
        Index adj_s = findValidNear(map, start);
        Index adj_g = findValidNear(map, goal);

        if (isBlocked(map, adj_s) || isBlocked(map, adj_g)) {
            ROS_ERROR("[AIS Planner] Start/Goal blocked or out of bounds!");
            return {};
        }

        std::priority_queue<Node*, std::vector<Node*>, NodeCmp> open;
        std::unordered_map<std::pair<int, int>, Node*, PairHash> all;

        // Guided A* uses reduced heuristic (0.1) for exploration;
        // Traditional A* uses 1.0 for optimal shortest-path
        double h_w = use_guidance ? 0.1 : 1.0;
        double h_init = h_w * (adj_g - adj_s).matrix().cast<double>().norm()
                        * map.getResolution();

        Node* s_node = new Node(adj_s.x(), adj_s.y(), 0.0, h_init);
        open.push(s_node);
        all[{adj_s.x(), adj_s.y()}] = s_node;

        int exp = 0;
        while (!open.empty() && exp++ < 300000) {
            Node* curr = open.top();
            open.pop();
            Index curr_idx(curr->x, curr->y);

            if ((curr_idx == adj_g).all()) {
                std::vector<Index> path;
                Node* p = curr;
                while (p) {
                    path.push_back(Index(p->x, p->y));
                    p = p->parent;
                }
                std::reverse(path.begin(), path.end());
                for (auto& n : all) delete n.second;
                ROS_INFO("[AIS Planner] %s Path found! Expanded %d nodes, "
                         "Path length: %zu",
                         use_guidance ? "[Lane-Guided]" : "[Traditional]",
                         exp, path.size());
                return path;
            }

            // 8-connected neighbourhood
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    Index nidx = curr_idx + Index(dx, dy);
                    if (isBlocked(map, nidx)) continue;

                    double ng = curr->g + stepCost(map, curr_idx, nidx,
                                                   use_guidance);
                    auto key = std::make_pair(nidx.x(), nidx.y());

                    if (all.find(key) == all.end() || ng < all[key]->g) {
                        double h = h_w
                                   * (adj_g - nidx).matrix().cast<double>().norm()
                                   * map.getResolution();
                        Node* n_node = new Node(nidx.x(), nidx.y(), ng, h, curr);
                        if (all.count(key)) delete all[key];
                        all[key] = n_node;
                        open.push(n_node);
                    }
                }
            }
        }

        for (auto& n : all) delete n.second;
        ROS_WARN("[AIS Planner] No path found after %d expansions.", exp);
        return {};
    }

private:
    bool use_local_dir_;  // true = vector field, false = legacy scalar d
    int direction_;       // +1/-1 (legacy fallback)
    double lambda_;       // direction penalty weight (strong)
    double lambda_qual_;  // position quality weight (weak)
};

// ====================== Globals ======================

GridMap global_map({"phibar", "dir_x", "dir_y", "obstacle", "visualization"});
AISPlanner planner;
bool has_start = false, has_goal = false;
double g_sx, g_sy, g_gx, g_gy;
std::string guidance_mode = "lane";  // "lane" or "traditional_only"

// ====================== Helpers ======================

void sanitizeMap(GridMap& map) {
    for (auto& layer : map.getLayers()) {
        auto& data = map[layer];
        for (int i = 0; i < data.size(); ++i) {
            if (!std::isfinite(data(i))) data(i) = 0.0f;
            if (std::abs(data(i)) < 1e-10) data(i) = 0.0f;
        }
    }
}

/**
 * Load a CSV file into a named layer of the global GridMap.
 *
 * The CSV is stored as (csv_rows x csv_cols). To align with the ROS
 * coordinate convention (x right, y up), we transpose and reverse both
 * axes, matching the loading logic in the original waterway_map package.
 */
bool loadCSVToLayer(const std::string& path, const std::string& layer_name,
                    double res) {
    std::ifstream file(path);
    if (!file.is_open()) {
        ROS_ERROR("[AIS Planner] Cannot open CSV: %s", path.c_str());
        return false;
    }

    std::vector<std::vector<float>> data_2d;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<float> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stof(val));
            } catch (...) {
                // skip unparseable
            }
        }
        if (!row.empty()) data_2d.push_back(row);
    }

    if (data_2d.empty()) {
        ROS_ERROR("[AIS Planner] CSV is empty: %s", path.c_str());
        return false;
    }

    int csv_rows = data_2d.size();
    int csv_cols = data_2d[0].size();

    Eigen::MatrixXf mat_raw(csv_rows, csv_cols);
    for (int i = 0; i < csv_rows; ++i) {
        for (int j = 0; j < csv_cols; ++j) {
            mat_raw(i, j) = (j < (int)data_2d[i].size()) ? data_2d[i][j] : 0.0f;
        }
    }

    // Transpose + reverse: match ROS coordinate frame
    Eigen::MatrixXf mat = mat_raw.transpose().eval();
    mat = mat.rowwise().reverse().eval();
    mat = mat.colwise().reverse().eval();

    double lx = mat.rows() * res;
    double ly = mat.cols() * res;
    Position center(lx / 2.0, ly / 2.0);

    if (global_map.getSize()(0) == 0) {
        global_map.setGeometry(Length(lx, ly), res, center);
        global_map.setFrameId("map");
    }

    global_map[layer_name] = mat;
    sanitizeMap(global_map);

    ROS_INFO("[AIS Planner] Loaded '%s' layer: %.1f x %.1f m (%ld x %ld cells), "
             "range [%.3f, %.3f]",
             layer_name.c_str(), lx, ly, mat.rows(), mat.cols(),
             mat.minCoeff(), mat.maxCoeff());
    return true;
}

// ====================== Callbacks ======================

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    if (global_map.getSize()(0) == 0) return;
    for (int i = 0; i < msg->info.width; ++i) {
        for (int j = 0; j < msg->info.height; ++j) {
            double wx = msg->info.origin.position.x + (i + 0.5) * msg->info.resolution;
            double wy = msg->info.origin.position.y + (j + 0.5) * msg->info.resolution;
            Index idx;
            if (global_map.getIndex(Position(wx, wy), idx)) {
                global_map.at("obstacle", idx) =
                    (float)msg->data[j * msg->info.width + i];
            }
        }
    }
}

void startCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    g_sx = msg->pose.pose.position.x;
    g_sy = msg->pose.pose.position.y;
    has_start = true;
}

void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    g_gx = msg->pose.position.x;
    g_gy = msg->pose.position.y;
    has_goal = true;
}

// ====================== Main ======================

int main(int argc, char** argv) {
    ros::init(argc, argv, "ais_planner_node");
    ros::NodeHandle nh("~");

    // --- Parameters ---
    std::string phibar_csv_path;
    nh.param<std::string>("phibar_csv_path", phibar_csv_path,
        "/root/demon3.16/src/ais_navigation/map/navigation_potential_field.csv");

    std::string dir_x_csv_path;
    nh.param<std::string>("dir_x_csv_path", dir_x_csv_path,
        "/root/demon3.16/src/ais_navigation/map/dir_x.csv");
    std::string dir_y_csv_path;
    nh.param<std::string>("dir_y_csv_path", dir_y_csv_path,
        "/root/demon3.16/src/ais_navigation/map/dir_y.csv");

    double lane_lambda;
    nh.param<double>("lane_lambda", lane_lambda, 200.0);

    double lambda_qual;
    nh.param<double>("lambda_qual", lambda_qual, 1.0);

    double map_resolution;
    nh.param<double>("map_resolution", map_resolution, 0.5);

    bool use_lane_guidance;
    nh.param<bool>("use_lane_guidance", use_lane_guidance, true);

    bool use_local_dir;
    nh.param<bool>("use_local_dir", use_local_dir, true);

    // --- Publishers ---
    ros::Publisher map_pub =
        nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Publisher guided_path_pub =
        nh.advertise<nav_msgs::Path>("ais_planned_path", 1, true);
    ros::Publisher trad_path_pub =
        nh.advertise<nav_msgs::Path>("ais_traditional_path", 1, true);

    // --- Subscribers ---
    ros::Subscriber s_sub = nh.subscribe("/initialpose", 1, startCallback);
    ros::Subscriber g_sub = nh.subscribe("/move_base_simple/goal", 1, goalCallback);
    ros::Subscriber m_sub = nh.subscribe("/map", 1, mapCallback);

    // --- Load navigation potential field ---
    global_map.setFrameId("map");

    if (!loadCSVToLayer(phibar_csv_path, "phibar", map_resolution)) {
        ROS_ERROR("[AIS Planner] Failed to load phibar CSV!");
        return -1;
    }

    // Load expected direction vector field d(p) ∈ ℝ²
    bool has_dir_x = loadCSVToLayer(dir_x_csv_path, "dir_x", map_resolution);
    bool has_dir_y = loadCSVToLayer(dir_y_csv_path, "dir_y", map_resolution);
    bool has_vector_field = has_dir_x && has_dir_y;

    // Set visualization layer
    if (global_map.exists("phibar")) {
        global_map["visualization"] = global_map["phibar"].array().abs();
    }

    // Configure planner
    planner.setUseLocalDir(use_local_dir && has_vector_field);
    planner.setLambda(lane_lambda);

    ROS_INFO("[AIS Planner] Node started.");
    ROS_INFO("  Phibar CSV: %s", phibar_csv_path.c_str());
    ROS_INFO("  Vector field: dir_x=%s, dir_y=%s",
             has_dir_x ? "ok" : "missing", has_dir_y ? "ok" : "missing");
    ROS_INFO("  Mode: %s",
             use_local_dir && has_vector_field ?
             "vector-field d(p)" : "legacy scalar-d");
    ROS_INFO("  Lane lambda: %.2f", lane_lambda);
    ROS_INFO("  Topics: /ais_planned_path, /ais_traditional_path");

    // --- Main loop ---
    ros::Rate rate(5);
    while (ros::ok()) {
        // Update parameters dynamically
        nh.getParam("lane_lambda", lane_lambda);
        nh.getParam("lambda_qual", lambda_qual);
        nh.getParam("use_lane_guidance", use_lane_guidance);
        planner.setLambda(lane_lambda);
        planner.setLambdaQual(lambda_qual);

        if (has_start && has_goal) {
            Position ps(g_sx, g_sy), pg(g_gx, g_gy);
            Index si, gi;
            bool s_in = global_map.getIndex(ps, si);
            bool g_in = global_map.getIndex(pg, gi);

            if (s_in && g_in) {
                ROS_INFO("[AIS Planner] Request: Start(%.2f, %.2f) -> "
                         "Goal(%.2f, %.2f)",
                         ps.x(), ps.y(), pg.x(), pg.y());

                // Run lane-guided A* (with navigation potential field)
                if (use_lane_guidance) {
                    auto path_guided = planner.plan(global_map, si, gi, true);
                    if (!path_guided.empty()) {
                        nav_msgs::Path pmsg;
                        pmsg.header.frame_id = "map";
                        pmsg.header.stamp = ros::Time::now();
                        for (auto& idx : path_guided) {
                            Position p;
                            global_map.getPosition(idx, p);
                            geometry_msgs::PoseStamped ps_msg;
                            ps_msg.pose.position.x = p.x();
                            ps_msg.pose.position.y = p.y();
                            ps_msg.pose.position.z = 0.0;
                            ps_msg.pose.orientation.w = 1.0;
                            pmsg.poses.push_back(ps_msg);
                        }
                        guided_path_pub.publish(pmsg);
                        ROS_INFO("[AIS Planner] Lane-guided path published (%zu pts)",
                                 pmsg.poses.size());
                    }
                }

                // Run traditional A* (pure distance, no lane guidance)
                auto path_trad = planner.plan(global_map, si, gi, false);
                if (!path_trad.empty()) {
                    nav_msgs::Path pmsg;
                    pmsg.header.frame_id = "map";
                    pmsg.header.stamp = ros::Time::now();
                    for (auto& idx : path_trad) {
                        Position p;
                        global_map.getPosition(idx, p);
                        geometry_msgs::PoseStamped ps_msg;
                        ps_msg.pose.position.x = p.x();
                        ps_msg.pose.position.y = p.y();
                        ps_msg.pose.position.z = 0.0;
                        ps_msg.pose.orientation.w = 1.0;
                        pmsg.poses.push_back(ps_msg);
                    }
                    trad_path_pub.publish(pmsg);
                    ROS_INFO("[AIS Planner] Traditional path published (%zu pts)",
                             pmsg.poses.size());
                }
            } else {
                ROS_WARN("[AIS Planner] Point(s) outside map boundary. "
                         "Start in: %d, Goal in: %d", s_in, g_in);
            }
            has_start = has_goal = false;
        }

        // Publish grid map for RViz visualization
        grid_map_msgs::GridMap msg;
        GridMapRosConverter::toMessage(global_map, msg);
        map_pub.publish(msg);

        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
