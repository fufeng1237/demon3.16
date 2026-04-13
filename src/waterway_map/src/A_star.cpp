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

using namespace grid_map;

//================ A* Planner =================
struct Node {
    int x, y; double g, h; Node* parent;
    Node(int x_, int y_, double g_, double h_, Node* p = nullptr) : x(x_), y(y_), g(g_), h(h_), parent(p) {}
    double f() const { return g + h; }
};
struct NodeCmp { bool operator()(Node* a, Node* b) { return a->f() > b->f(); } };
struct PairHash { size_t operator()(const std::pair<int,int>& p) const { return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1); } };

class AStarPlanner {
public:
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

    // [修改点 1]：增加 use_guidance 参数
    double stepCost(const GridMap& map, const Index& cur, const Index& next, bool use_guidance) {
        double dist = (next - cur).matrix().cast<double>().norm() * map.getResolution();
        
        // 如果是传统 A*，直接返回真实物理距离
        if (!use_guidance) {
            return dist;
        }
        
        // 如果是引导 A*，叠加奖励场权重
        Position p_cur, p_next;
        map.getPosition(cur, p_cur);
        map.getPosition(next, p_next);
        double dx_world = p_next.x() - p_cur.x(); 
        
        float reward = map.at("combined_field", next);
        double alignment = dx_world * static_cast<double>(reward);
        double multiplier = 1.0 - 0.005 * alignment;
        
        return dist * std::max(0.1, std::min(2.0, multiplier));
    }

    // [修改点 2]：增加 use_guidance 参数并透传给 stepCost
    std::vector<Index> plan(const GridMap& map, Index start, Index goal, bool use_guidance) {
        Index adj_s = findValidNear(map, start);
        Index adj_g = findValidNear(map, goal);

        if (isBlocked(map, adj_s) || isBlocked(map, adj_g)) {
            ROS_ERROR("A* Aborted: Adjusted Start/Goal is BLOCKED or OUT of map bounds!");
            return {};
        }

        std::priority_queue<Node*, std::vector<Node*>, NodeCmp> open;
        std::unordered_map<std::pair<int,int>, Node*, PairHash> all;

        // 传统A*权重用1.0保证最短路径，引导A*维持0.1保证探索性
        double h_w = use_guidance ? 0.1 : 1.0; 
        double h_init = h_w * (adj_g - adj_s).matrix().cast<double>().norm() * map.getResolution();
        
        Node* s_node = new Node(adj_s.x(), adj_s.y(), 0.0, h_init);
        open.push(s_node);
        all[{adj_s.x(), adj_s.y()}] = s_node;

        int exp = 0;
        while (!open.empty() && exp++ < 300000) {
            Node* curr = open.top(); open.pop();
            Index curr_idx(curr->x, curr->y);

            if ((curr_idx == adj_g).all()) {
                std::vector<Index> path; Node* p = curr;
                while(p) { path.push_back(Index(p->x, p->y)); p = p->parent; }
                std::reverse(path.begin(), path.end());
                for(auto& n : all) delete n.second; 
                ROS_INFO("%s Path found! Expanded %d nodes.", use_guidance ? "[Guided]" : "[Traditional]", exp);
                return path;
            }

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    Index nidx = curr_idx + Index(dx, dy);
                    if (isBlocked(map, nidx)) continue;

                    double ng = curr->g + stepCost(map, curr_idx, nidx, use_guidance);
                    auto key = std::make_pair(nidx.x(), nidx.y());

                    if (all.find(key) == all.end() || ng < all[key]->g) {
                        double h = h_w * (adj_g - nidx).matrix().cast<double>().norm() * map.getResolution();
                        Node* n_node = new Node(nidx.x(), nidx.y(), ng, h, curr);
                        if (all.count(key)) delete all[key];
                        all[key] = n_node; open.push(n_node);
                    }
                }
            }
        }
        for(auto& n : all) delete n.second;
        ROS_WARN("Search finished. No path available.");
        return {};
    }
};

//================ Global & Loaders =================
GridMap global_map({"combined_field", "obstacle", "visualization"});
AStarPlanner planner;
bool has_start = false, has_goal = false;
double g_sx, g_sy, g_gx, g_gy;

void sanitizeMap(GridMap& map) {
    for (auto& layer : map.getLayers()) {
        auto& data = map[layer];
        for (int i = 0; i < data.size(); ++i) {
            if (!std::isfinite(data(i))) data(i) = 0.0f;
            if (std::abs(data(i)) < 1e-10) data(i) = 0.0f;
        }
    }
}

// ... (loadCSV 和 mapCallback, startCallback, goalCallback 保持原样)
bool loadCSV(const std::string& path, double res) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::vector<std::vector<float>> data_2d;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<float> row; std::stringstream ss(line); std::string val;
        while (std::getline(ss, val, ',')) { try { row.push_back(std::stof(val)); } catch (...) {} }
        if (!row.empty()) data_2d.push_back(row);
    }
    
    int csv_rows = data_2d.size();    
    int csv_cols = data_2d[0].size(); 
    Eigen::MatrixXf mat_raw(csv_rows, csv_cols);
    for (int i = 0; i < csv_rows; ++i)
        for (int j = 0; j < csv_cols; ++j)
            mat_raw(i, j) = (j < data_2d[i].size()) ? data_2d[i][j] : 0.0f;

    Eigen::MatrixXf mat = mat_raw.transpose().eval();
    mat = mat.rowwise().reverse().eval(); 
    mat = mat.colwise().reverse().eval();

    double lx = mat.rows() * res; 
    double ly = mat.cols() * res; 
    
    Position center(lx / 2.0, ly / 2.0);
    global_map.setGeometry(Length(lx, ly), res, center);
    global_map["combined_field"] = mat;
    global_map["visualization"] = mat.array().abs();
    
    sanitizeMap(global_map);
    ROS_INFO("Map Loaded: %.1f x %.1f m (%ld x %ld cells). Center: [%.1f, %.1f]", 
             lx, ly, mat.rows(), mat.cols(), center.x(), center.y());
    return true;
}

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    if (global_map.getSize()(0) == 0) return;
    for (int i = 0; i < msg->info.width; ++i) {
        for (int j = 0; j < msg->info.height; ++j) {
            double wx = msg->info.origin.position.x + (i + 0.5) * msg->info.resolution;
            double wy = msg->info.origin.position.y + (j + 0.5) * msg->info.resolution;
            Index idx;
            if (global_map.getIndex(Position(wx, wy), idx)) {
                global_map.at("obstacle", idx) = (float)msg->data[j * msg->info.width + i];
            }
        }
    }
}

void startCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    g_sx = msg->pose.pose.position.x; g_sy = msg->pose.pose.position.y; has_start = true;
}
void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    g_gx = msg->pose.position.x; g_gy = msg->pose.position.y; has_goal = true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "planner_master_node");
    ros::NodeHandle nh("~");

    ros::Publisher map_pub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    
    // [修改点 3]：注册两个 Publisher
    ros::Publisher guided_path_pub = nh.advertise<nav_msgs::Path>("planned_path", 1, true); // 引导版 (继续被优化器使用)
    ros::Publisher trad_path_pub = nh.advertise<nav_msgs::Path>("traditional_path", 1, true); // 传统版

    ros::Subscriber s_sub = nh.subscribe("/initialpose", 1, startCallback);
    ros::Subscriber g_sub = nh.subscribe("/move_base_simple/goal", 1, goalCallback);
    ros::Subscriber m_sub = nh.subscribe("/map", 1, mapCallback);

    std::string csv_path;
    nh.param<std::string>("csv_path", csv_path, "/home/ros_ws/demon3.16/src/waterway_map/map/combined_distance_field.csv");
    global_map.setFrameId("map");

    if (!loadCSV(csv_path,0.5 )) { ROS_ERROR("CSV Load Failed"); return -1; }

    ros::Rate rate(5); 
    while (ros::ok()) {
        if (has_start && has_goal) {
            Position ps(g_sx, g_sy), pg(g_gx, g_gy);
            Index si, gi;
            bool s_in = global_map.getIndex(ps, si);
            bool g_in = global_map.getIndex(pg, gi);

            if (s_in && g_in) {
                ROS_INFO("Request: Start(%.2f, %.2f) -> Goal(%.2f, %.2f)", ps.x(), ps.y(), pg.x(), pg.y());
                
                // [修改点 4]：一次请求，跑两次 A*
                auto path_guided = planner.plan(global_map, si, gi, true);  // 跑引导式 A*
                auto path_trad = planner.plan(global_map, si, gi, false); // 跑传统 A*
                
                // 发布引导式 A* 路径
                if (!path_guided.empty()) {
                    nav_msgs::Path pmsg; pmsg.header.frame_id = "map"; pmsg.header.stamp = ros::Time::now();
                    for (auto& idx : path_guided) {
                        Position p; global_map.getPosition(idx, p);
                        geometry_msgs::PoseStamped ps_msg; ps_msg.pose.position.x = p.x(); ps_msg.pose.position.y = p.y();
                        pmsg.poses.push_back(ps_msg);
                    }
                    guided_path_pub.publish(pmsg);
                }

                // 发布传统 A* 路径
                if (!path_trad.empty()) {
                    nav_msgs::Path pmsg; pmsg.header.frame_id = "map"; pmsg.header.stamp = ros::Time::now();
                    for (auto& idx : path_trad) {
                        Position p; global_map.getPosition(idx, p);
                        geometry_msgs::PoseStamped ps_msg; ps_msg.pose.position.x = p.x(); ps_msg.pose.position.y = p.y();
                        pmsg.poses.push_back(ps_msg);
                    }
                    trad_path_pub.publish(pmsg);
                }

            } else {
                ROS_WARN("Click rejected: Point(s) outside boundary. Start in: %d, Goal in: %d", s_in, g_in);
            }
            has_start = has_goal = false;
        }
        
        // 发布地图逻辑...
        // ... (保持不变)
        grid_map_msgs::GridMap msg;
        GridMapRosConverter::toMessage(global_map, msg);
        map_pub.publish(msg);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}