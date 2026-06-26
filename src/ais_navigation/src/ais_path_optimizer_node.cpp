/**
 * ais_path_optimizer_node.cpp
 * Path smoother with lane deviation penalty from AIS navigation potential field.
 *
 * Method (ref: 基于AIS导航势场的无人艇路径规划.md §4.3):
 *
 *   Minimize E(P) = omega_d * SUM ||p_i - p_i_raw||^2       (data fidelity)
 *                 + omega_s * SUM ||p_{i-1} - 2p_i + p_{i+1}||^2  (smoothness)
 *                 + omega_v * SUM c_lane(p_i; d)             (lane deviation)
 *
 *   where c_lane(p; d) = max(0, -d * Phi_bar(p))
 *
 *   Gradient of lane cost (Equation 7):
 *     grad(c_lane) = -d * grad(Phi_bar)   if -d*Phi_bar > 0  (wrong lane)
 *                  = 0                     otherwise
 *
 * The lane deviation force replaces the geometric "safe corridor" penalty
 * from the original waterway_map package.
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using namespace grid_map;

class AISPathOptimizer {
public:
    AISPathOptimizer(ros::NodeHandle& nh) {
        // ---- Smoothing weights ----
        nh.param("weight_data", weight_data_, 0.2);
        nh.param("weight_smooth", weight_smooth_, 0.5);

        // ---- Lane deviation parameters ----
        nh.param("weight_lane", weight_lane_, 0.1);
        nh.param("lane_direction", lane_direction_, 1);

        // ---- Iteration control ----
        nh.param("tolerance", tolerance_, 0.0001);
        nh.param("max_iterations", max_iterations_, 1000);

        // ---- Map loading ----
        std::string phibar_csv;
        nh.param<std::string>("phibar_csv_path", phibar_csv,
            "/root/demon3.16/src/ais_navigation/map/navigation_potential_field.csv");
        std::string grad_x_csv;
        nh.param<std::string>("grad_x_csv_path", grad_x_csv,
            "/root/demon3.16/src/ais_navigation/map/phibar_grad_x.csv");
        std::string grad_y_csv;
        nh.param<std::string>("grad_y_csv_path", grad_y_csv,
            "/root/demon3.16/src/ais_navigation/map/phibar_grad_y.csv");
        nh.param("map_resolution", map_resolution_, 0.5);

        // Pre-register layers before loading CSV data
        lane_map_.add("phibar");
        lane_map_.add("phibar_grad_x");
        lane_map_.add("phibar_grad_y");

        loadMapLayers(phibar_csv, grad_x_csv, grad_y_csv, map_resolution_);

        // ---- ROS I/O ----
        path_sub_ = nh.subscribe("planned_path", 1,
                                 &AISPathOptimizer::pathCallback, this);
        path_pub_ = nh.advertise<nav_msgs::Path>("smoothed_path", 1);

        ROS_INFO("[AIS Optimizer] Node started with lane deviation optimization.");
        ROS_INFO("  weight_data=%.2f, weight_smooth=%.2f, weight_lane=%.2f",
                 weight_data_, weight_smooth_, weight_lane_);
        ROS_INFO("  lane_direction=%d (%s), tolerance=%.6f",
                 lane_direction_,
                 lane_direction_ == 1 ? "downstream" : "upstream",
                 tolerance_);
    }

private:
    ros::Subscriber path_sub_;
    ros::Publisher path_pub_;

    double weight_data_;
    double weight_smooth_;
    double weight_lane_;
    int lane_direction_;
    double tolerance_;
    int max_iterations_;
    double map_resolution_;

    // Local grid map holding phibar and gradient layers
    GridMap lane_map_;

    struct Point2D {
        double x, y;
        Point2D() : x(0), y(0) {}
        Point2D(double x_, double y_) : x(x_), y(y_) {}
    };

    // ---- Map loading ----

    bool loadCSVToLayer(const std::string& path, const std::string& layer_name,
                        double res) {
        std::ifstream file(path);
        if (!file.is_open()) {
            ROS_WARN("[AIS Optimizer] Cannot open CSV: %s (layer '%s' will be "
                     "unavailable)", path.c_str(), layer_name.c_str());
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
                } catch (...) {}
            }
            if (!row.empty()) data_2d.push_back(row);
        }

        if (data_2d.empty()) return false;

        int csv_rows = data_2d.size();
        int csv_cols = data_2d[0].size();

        Eigen::MatrixXf mat_raw(csv_rows, csv_cols);
        for (int i = 0; i < csv_rows; ++i)
            for (int j = 0; j < csv_cols; ++j)
                mat_raw(i, j) = (j < (int)data_2d[i].size()) ? data_2d[i][j] : 0.0f;

        Eigen::MatrixXf mat = mat_raw.transpose().eval();
        mat = mat.rowwise().reverse().eval();
        mat = mat.colwise().reverse().eval();

        double lx = mat.rows() * res;
        double ly = mat.cols() * res;

        if (lane_map_.getSize()(0) == 0) {
            lane_map_.setGeometry(Length(lx, ly), res, Position(lx / 2.0, ly / 2.0));
            lane_map_.setFrameId("map");
        }

        lane_map_[layer_name] = mat;

        ROS_INFO("[AIS Optimizer] Loaded '%s' layer: %ldx%ld, range [%.4f, %.4f]",
                 layer_name.c_str(), mat.rows(), mat.cols(),
                 mat.minCoeff(), mat.maxCoeff());
        return true;
    }

    void loadMapLayers(const std::string& phibar_path,
                       const std::string& grad_x_path,
                       const std::string& grad_y_path, double res) {
        bool ok = loadCSVToLayer(phibar_path, "phibar", res);
        loadCSVToLayer(grad_x_path, "phibar_grad_x", res);
        loadCSVToLayer(grad_y_path, "phibar_grad_y", res);

        if (!ok) {
            ROS_WARN("[AIS Optimizer] Phibar layer not loaded. Lane deviation "
                     "forces will be zero.");
        }
    }

    // ---- Lane compliance cost and gradient ----

    double getPhibarAtPosition(double wx, double wy) const {
        if (!lane_map_.exists("phibar")) return 0.0;
        Index idx;
        if (!lane_map_.getIndex(Position(wx, wy), idx)) return 0.0;
        if (idx(0) < 0 || idx(0) >= lane_map_.getSize()(0) ||
            idx(1) < 0 || idx(1) >= lane_map_.getSize()(1))
            return 0.0;
        return lane_map_.at("phibar", idx);
    }

    double getGradXAtPosition(double wx, double wy) const {
        if (!lane_map_.exists("phibar_grad_x")) return 0.0;
        Index idx;
        if (!lane_map_.getIndex(Position(wx, wy), idx)) return 0.0;
        if (idx(0) < 0 || idx(0) >= lane_map_.getSize()(0) ||
            idx(1) < 0 || idx(1) >= lane_map_.getSize()(1))
            return 0.0;
        return lane_map_.at("phibar_grad_x", idx);
    }

    double getGradYAtPosition(double wx, double wy) const {
        if (!lane_map_.exists("phibar_grad_y")) return 0.0;
        Index idx;
        if (!lane_map_.getIndex(Position(wx, wy), idx)) return 0.0;
        if (idx(0) < 0 || idx(0) >= lane_map_.getSize()(0) ||
            idx(1) < 0 || idx(1) >= lane_map_.getSize()(1))
            return 0.0;
        return lane_map_.at("phibar_grad_y", idx);
    }

    // ---- Path callback ----

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        if (msg->poses.size() < 3) {
            path_pub_.publish(*msg);
            return;
        }

        // Update parameters dynamically
        ros::NodeHandle nh("~");
        nh.getParam("weight_data", weight_data_);
        nh.getParam("weight_smooth", weight_smooth_);
        nh.getParam("weight_lane", weight_lane_);
        nh.getParam("lane_direction", lane_direction_);

        std::vector<Point2D> raw_path;
        for (const auto& pose : msg->poses) {
            raw_path.push_back(Point2D(pose.pose.position.x,
                                       pose.pose.position.y));
        }

        std::vector<Point2D> smoothed = smoothPath(raw_path);

        nav_msgs::Path out_msg;
        out_msg.header = msg->header;
        out_msg.header.stamp = ros::Time::now();
        for (size_t i = 0; i < smoothed.size(); ++i) {
            geometry_msgs::PoseStamped ps;
            ps.header = out_msg.header;
            if (i < msg->poses.size()) {
                ps.pose = msg->poses[i].pose;
            }
            ps.pose.position.x = smoothed[i].x;
            ps.pose.position.y = smoothed[i].y;
            out_msg.poses.push_back(ps);
        }

        path_pub_.publish(out_msg);
        ROS_INFO("[AIS Optimizer] Path smoothed: %zu points", raw_path.size());
    }

    // ---- Core smoothing algorithm ----

    std::vector<Point2D> smoothPath(const std::vector<Point2D>& raw_path) {
        std::vector<Point2D> new_path = raw_path;
        double change = tolerance_;
        int iterations = 0;
        int direction = lane_direction_;

        while (change >= tolerance_ && iterations < max_iterations_) {
            change = 0.0;

            // Keep first and last points fixed
            for (size_t i = 1; i < raw_path.size() - 1; ++i) {
                double old_x = new_path[i].x;
                double old_y = new_path[i].y;

                // ---- 1. Data fidelity force (pull toward raw A* path) ----
                double data_fx = weight_data_ * (raw_path[i].x - old_x);
                double data_fy = weight_data_ * (raw_path[i].y - old_y);

                // ---- 2. Smoothing force (Laplacian / 2nd order) ----
                double smooth_fx = weight_smooth_ *
                    (new_path[i - 1].x + new_path[i + 1].x - 2.0 * old_x);
                double smooth_fy = weight_smooth_ *
                    (new_path[i - 1].y + new_path[i + 1].y - 2.0 * old_y);

                // ---- 3. Lane deviation force (replaces corridor penalty) ----
                double lane_fx = 0.0;
                double lane_fy = 0.0;

                double phibar = getPhibarAtPosition(old_x, old_y);
                double lane_cost = std::max(0.0, -direction * phibar);

                // Only apply force when in the WRONG lane (c_lane > 0)
                if (lane_cost > 0.0) {
                    double grad_x = getGradXAtPosition(old_x, old_y);
                    double grad_y = getGradYAtPosition(old_x, old_y);

                    // Equation (7): grad(c_lane) = -d * grad(Phi_bar)
                    // Force pushes point along -grad(c_lane) direction
                    // i.e., toward correct lane (where c_lane decreases)
                    lane_fx = -weight_lane_ * direction * grad_x;
                    lane_fy = -weight_lane_ * direction * grad_y;
                }

                // ---- Combined update ----
                new_path[i].x = old_x + data_fx + smooth_fx + lane_fx;
                new_path[i].y = old_y + data_fy + smooth_fy + lane_fy;

                change += std::abs(old_x - new_path[i].x) +
                          std::abs(old_y - new_path[i].y);
            }
            iterations++;
        }

        ROS_DEBUG("[AIS Optimizer] Smoothing converged in %d iterations "
                  "(final change=%.6f)", iterations, change);
        return new_path;
    }
};

// ====================== Main ======================

int main(int argc, char** argv) {
    ros::init(argc, argv, "ais_path_optimizer_node");
    ros::NodeHandle nh("~");

    AISPathOptimizer optimizer(nh);

    ros::spin();
    return 0;
}
