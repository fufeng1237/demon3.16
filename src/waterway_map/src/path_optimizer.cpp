#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <cmath>

class PathOptimizerNode {
public:
    PathOptimizerNode(ros::NodeHandle& nh) {
        // 读取平滑参数 (原有)
        nh.param("weight_data", weight_data_, 0.9);   // 保持原路径特征的权重
        nh.param("weight_smooth", weight_smooth_, 0.5); // 平滑的权重
        
        // --- 新增：航道偏离代价参数 ---
        nh.param("weight_deviation", weight_deviation_, 0.8); // 偏离代价权重 (通常设得比较大，起强制约束作用)
        nh.param("safe_corridor", safe_corridor_, 1.0);       // 允许的最大安全偏离距离(米)，超过此阈值即触发惩罚

        // 迭代控制 (原有)
        nh.param("tolerance", tolerance_, 0.0001);    
        nh.param("max_iterations", max_iterations_, 1000); 

        // 订阅原始路径，发布平滑后路径
        // 使用私有命名空间 topic，支持 launch 文件中 remap
        path_sub_ = nh.subscribe("planned_path", 1, &PathOptimizerNode::pathCallback, this);
        path_pub_ = nh.advertise<nav_msgs::Path>("smoothed_path", 1);
        
        ROS_INFO("Path Optimizer Node started with Channel Deviation Penalty.");
    }

private:
    ros::Subscriber path_sub_;
    ros::Publisher path_pub_;

    double weight_data_;
    double weight_smooth_;
    double weight_deviation_; // 新增偏离权重
    double safe_corridor_;    // 新增安全走廊宽度
    double tolerance_;
    int max_iterations_;

    struct Point2D {
        double x, y;
    };

    void pathCallback(const nav_msgs::Path::ConstPtr& msg) {
        if (msg->poses.size() < 3) {
            path_pub_.publish(*msg);
            return;
        }

        std::vector<Point2D> raw_path;
        for (const auto& pose : msg->poses) {
            raw_path.push_back({pose.pose.position.x, pose.pose.position.y});
        }

        std::vector<Point2D> smoothed_path = smoothPath(raw_path);

        nav_msgs::Path out_msg;
        out_msg.header = msg->header;
        out_msg.header.stamp = ros::Time::now(); // 刷新时间戳确保 RViz 正常显示

        for (int i = 0; i < smoothed_path.size(); ++i) {
            geometry_msgs::PoseStamped new_pose = msg->poses[i]; 
            new_pose.pose.position.x = smoothed_path[i].x;
            new_pose.pose.position.y = smoothed_path[i].y;
            out_msg.poses.push_back(new_pose);
        }

        path_pub_.publish(out_msg);
        ROS_INFO("Path smoothed. Points: %zu", raw_path.size());
    }

    // 核心平滑算法 (加入航道偏离代价)
    std::vector<Point2D> smoothPath(const std::vector<Point2D>& raw_path) {
        std::vector<Point2D> new_path = raw_path;
        double change = tolerance_;
        int iterations = 0;

        while (change >= tolerance_ && iterations < max_iterations_) {
            change = 0.0;
            // 首尾两点保持不动，只优化中间的点
            for (size_t i = 1; i < raw_path.size() - 1; ++i) {
                
                double old_x = new_path[i].x;
                double old_y = new_path[i].y;

                // 1. 数据保真力 (拉向原始 A* 路径点)
                double data_force_x = weight_data_ * (raw_path[i].x - old_x);
                double data_force_y = weight_data_ * (raw_path[i].y - old_y);

                // 2. 平滑力 (拉向相邻两点的中心点)
                double smooth_force_x = weight_smooth_ * (new_path[i-1].x + new_path[i+1].x - 2.0 * old_x);
                double smooth_force_y = weight_smooth_ * (new_path[i-1].y + new_path[i+1].y - 2.0 * old_y);

                // 3. --- 航道偏离代价力 ---
                double dev_force_x = 0.0;
                double dev_force_y = 0.0;
                
                // 计算当前点距离原始安全路径的偏差向量和距离
                double dx = raw_path[i].x - old_x;
                double dy = raw_path[i].y - old_y;
                double distance = std::sqrt(dx * dx + dy * dy);

                // 如果为了平滑导致偏离超过了设定的"安全走廊"宽度，则施加强烈的惩罚力将其拽回
                if (distance > safe_corridor_ && distance > 0.00001) {
                    // 偏离越多，拉回的惩罚系数越大
                    double penalty = (distance - safe_corridor_) / distance; 
                    dev_force_x = weight_deviation_ * dx * penalty;
                    dev_force_y = weight_deviation_ * dy * penalty;
                }

                // 综合受力，更新位置
                new_path[i].x = old_x + data_force_x + smooth_force_x + dev_force_x;
                new_path[i].y = old_y + data_force_y + smooth_force_y + dev_force_y;

                change += std::abs(old_x - new_path[i].x) + std::abs(old_y - new_path[i].y);
            }
            iterations++;
        }
        
        ROS_DEBUG("Smoothing finished in %d iterations.", iterations);
        return new_path;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_optimizer_node");
    ros::NodeHandle nh("~");
    
    PathOptimizerNode optimizer(nh);
    
    ros::spin();
    return 0;
}