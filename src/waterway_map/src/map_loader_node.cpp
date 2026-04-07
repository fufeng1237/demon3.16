#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <sensor_msgs/PointCloud2.h>
#include <fstream>

using namespace grid_map;

bool loadCSVToGridMap(const std::string& filename, const std::string& layer_name, 
                      GridMap& map, double resolution, float& min_val, float& max_val) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        ROS_ERROR("Failed to open file: %s", filename.c_str());
        return false;
    }

    std::vector<std::vector<float>> data_2d;
    std::string line;
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                float fval = std::stof(val);
                row.push_back(fval);
                if (fval < min_val) min_val = fval;
                if (fval > max_val) max_val = fval;
            } catch (...) { row.push_back(0.0f); }
        }
        if (!row.empty()) data_2d.push_back(row);
    }

    if (data_2d.empty()) return false;

    // --- 核心修复：处理行列转置 ---
    int csv_rows = data_2d.size();    // 90 (对应 Y)
    int csv_cols = data_2d[0].size(); // 180 (对应 X)
    
    // 1. 先按 CSV 原始维度填充（90x180）
    Eigen::MatrixXf mat_raw(csv_rows, csv_cols);
    for (int i = 0; i < csv_rows; ++i) {
        for (int j = 0; j < csv_cols; ++j) {
            mat_raw(i, j) = (j < data_2d[i].size()) ? data_2d[i][j] : 0.0f;
        }
    }

    // 2. 转置矩阵：变为 180x90，使其匹配 GridMap 的 (X, Y) 映射
    // 现在 Rows = 180 (X), Cols = 90 (Y)
    Eigen::MatrixXf mat = mat_raw.transpose().eval();

    // 3. 修正方向：通常转置后需要水平翻转来匹配坐标系
    mat = mat.rowwise().reverse().eval(); 
    mat = mat.colwise().reverse().eval();

    // 4. 设置物理尺寸
    double lx = mat.rows() * resolution; // 180 * 0.5 = 90.0m
    double ly = mat.cols() * resolution; // 90 * 0.5 = 45.0m
    
    // Position 为中心点：原点在(0,0)，中心就在(lx/2, ly/2)
    Position center(lx / 2.0, ly / 2.0);
    map.setGeometry(Length(lx, ly), resolution, center);

    // 5. 写入数据（此时 mat 的尺寸 180x90 完美匹配 map.getSize()）
    map[layer_name] = mat;

    if (max_val > min_val) {
        map[layer_name + "_normalized"] = (mat.array() - min_val) / (max_val - min_val);
        map[layer_name + "_color"] = map[layer_name + "_normalized"] * 255.0f;
    }

    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "waterway_map_loader");
    ros::NodeHandle nh("~");
    
    // CRITICAL: Publisher 必须放在循环外面！
    ros::Publisher publisher = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("elevation_cloud", 1);
    
    double resolution = 0.5;
    std::string csv_path;
    nh.param<std::string>("csv_path", csv_path, "/home/fufeng/map/combined_distance_field3.csv");
    
    // 初始化所有图层
    GridMap map({"combined_field", "combined_field_normalized", "combined_field_color", "binary_lanes", "elevation"});
    map.setFrameId("map");
    
    float min_val, max_val;
    if (!loadCSVToGridMap(csv_path, "combined_field", map, resolution, min_val, max_val)) {
        ROS_ERROR("Failed to load map.");
        return -1;
    }

    // 在循环外计算一次辅助图层，避免重复计算
    for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const Index index(*it);
        float val = map.at("combined_field", index);
        map.at("binary_lanes", index) = (val > 0) ? 1.0f : ((val < 0) ? -1.0f : 0.0f);
        if (map.exists("combined_field_normalized")) {
            map.at("elevation", index) = map.at("combined_field_normalized", index) * 5.0f;
        }
    }

    ros::Rate rate(1.0); 
    while (ros::ok()) {
        map.setTimestamp(ros::Time::now().toNSec());
        
        grid_map_msgs::GridMap message;
        GridMapRosConverter::toMessage(map, message);
        publisher.publish(message);
        
        if (map.exists("elevation")) {
            sensor_msgs::PointCloud2 pointcloud;
            GridMapRosConverter::toPointCloud(map, "elevation", pointcloud);
            pc_pub.publish(pointcloud);
        }
        
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}