#ifndef TASK_PLANNER_COMMON_H
#define TASK_PLANNER_COMMON_H

#include <string>
#include <vector>
#include <unordered_map>

// USV（无人船）结构
struct USV {
    int id;
    int x;
    int y;
    double max_speed;
    int capacity;                    // 最大承重（重量）
    int volume_capacity;             // 最大体积容量
    double energy;                   // 当前能源
    double max_energy;               // 最大能源容量
    double base_energy_consumption;  // 基础能耗系数
    double payload_factor;           // 负重能耗因子（每单位负重增加的能耗）
};

// 港口结构
struct Port {
    int id;
    int x;
    int y;
};

// 加油站结构
struct GasStation {
    int id;
    int x;
    int y;
    int node_id;                 // 关联的路网节点ID
    double refuel_rate;          // 加油速率
};

// 文件解析函数声明
std::vector<USV> parseUSVFile(const std::string& filepath);
std::unordered_map<int, Port> parsePortFile(const std::string& filepath);
std::vector<GasStation> parseGasStationFile(const std::string& filepath);

#endif // USV_ROADMAP_COMMON_H
