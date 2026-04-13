#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace grid_map;

class HeatmapLoader {
public:
    HeatmapLoader() : nh_("~") {
        // 1. 读取参数
        nh_.param<std::string>("csv_path", csv_path_, "");
        nh_.param<std::string>("frame_id", frame_id_, "map");
        nh_.param<std::string>("layer_name", layer_name_, "combined_field");
        
        // 分辨率和地图总物理长度参数
        nh_.param<double>("resolution", res_param_, 0.5);
        nh_.param<double>("map_total_length_x", map_lx_, 0.0); 
        nh_.param<double>("map_total_length_y", map_ly_, 0.0);

        publisher_ = nh_.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);

        if (loadAndPublish()) {
            ROS_INFO("Heatmap successfully loaded and aligned!");
        } else {
            ROS_ERROR("Failed to load heatmap from CSV.");
        }
    }

    bool loadAndPublish() {
        // 2. 读取 CSV 文件
        std::ifstream file(csv_path_);
        if (!file.is_open()) {
            ROS_ERROR("Cannot open CSV file at: %s", csv_path_.c_str());
            return false;
        }

        std::vector<std::vector<float>> data_2d;
        std::string line;
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                try { row.push_back(std::stof(val)); }
                catch (...) { row.push_back(0.0f); }
            }
            if (!row.empty()) data_2d.push_back(row);
        }

        if (data_2d.empty()) {
            ROS_ERROR("CSV file is empty or invalid.");
            return false;
        }

        // 3. 获取 CSV 数据的行数和列数
        int csv_rows = data_2d.size();      // 对应 Y 轴 (高度)
        int csv_cols = data_2d[0].size();   // 对应 X 轴 (宽度)

        // 4. 核心对齐逻辑：计算物理总长度
        // 如果 Launch 文件没有提供总长度，则默认按 (CSV列数 * 设定的分辨率) 计算
        if (map_lx_ <= 0) map_lx_ = csv_cols * res_param_;
        if (map_ly_ <= 0) map_ly_ = csv_rows * res_param_;

        // 根据物理总长度和 CSV 数据网格数，重新计算内部自适应分辨率
        // 这能保证无论 CSV 有多少个数据点，都会被完美拉伸到覆盖整个栅格地图
        double internal_res = map_lx_ / csv_cols; 

        // 5. 配置 GridMap 几何参数
        map_.setFrameId(frame_id_);
        // 中心点设为 (Lx/2, Ly/2) 确保地图的左下角刚好对齐坐标原点 (0,0)
        map_.setGeometry(Length(map_lx_, map_ly_), internal_res, Position(map_lx_ / 2.0, map_ly_ / 2.0));
        map_.add(layer_name_);

        // 6. 填充热力图数据
        Matrix& data = map_[layer_name_];
        for (int i = 0; i < csv_rows; ++i) {
            for (int j = 0; j < csv_cols; ++j) {
                // ROS 坐标系通常与图像坐标系上下相反
                // 如果发现图像在 RViz 中上下颠倒，请使用 data_2d[csv_rows - 1 - i][j]
                Index index(i, j);
                if (map_.isValid(index)) {
                    data(index(0), index(1)) = data_2d[csv_rows - 1 - i][j]; // 默认加入上下翻转修正
                }
            }
        }

        // 打印对齐信息供检查
        ROS_INFO("=== Alignment Info ===");
        ROS_INFO("CSV Grid Size: %d (width) x %d (height)", csv_cols, csv_rows);
        ROS_INFO("Physical Size: %.2f m x %.2f m", map_lx_, map_ly_);
        ROS_INFO("Adaptive Res : %.8f m/cell", internal_res);
        ROS_INFO("======================");

        // 7. 进入发布循环
        ros::Rate rate(1.0); // 1Hz 发布频率足以在 RViz 中显示静态地图
        while (ros::ok()) {
            map_.setTimestamp(ros::Time::now().toNSec());
            grid_map_msgs::GridMap msg;
            GridMapRosConverter::toMessage(map_, msg);
            publisher_.publish(msg);
            
            ros::spinOnce();
            rate.sleep();
        }
        return true;
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher publisher_;
    GridMap map_;
    std::string csv_path_;
    std::string frame_id_;
    std::string layer_name_;
    double map_lx_, map_ly_, res_param_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "heatmap_loader");
    HeatmapLoader loader;
    return 0;
}