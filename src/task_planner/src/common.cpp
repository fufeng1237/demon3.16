#include "task_planner/common.h"
#include <iostream>
#include <fstream>
#include <regex>

std::vector<USV> parseUSVFile(const std::string& filepath) {
    std::vector<USV> usvs;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开 USV 文件: " << filepath << std::endl;
        return usvs;
    }

    std::string line;
    // 新格式：USV: ID, X, Y, MaxSpeed, WeightCapacity, VolumeCapacity, Energy, MaxEnergy, BaseConsumption, PayloadFactor
    std::regex usv_new_regex(R"(USV:\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+))");
    // 旧格式：USV_ID: ID, Position: (X, Y), MaxSpeed: S, Capacity: C
    std::regex usv_old_regex(R"(USV_ID:\s*(\d+),\s*Position:\s*\((\d+),\s*(\d+)\),\s*MaxSpeed:\s*([\d.]+),\s*Capacity:\s*(\d+))");
    std::smatch match;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // 先尝试新格式
        if (std::regex_search(line, match, usv_new_regex)) {
            USV usv;
            usv.id = std::stoi(match[1].str());
            usv.x = std::stoi(match[2].str());
            usv.y = std::stoi(match[3].str());
            usv.max_speed = std::stod(match[4].str());
            usv.capacity = std::stoi(match[5].str());
            usv.volume_capacity = std::stoi(match[6].str());
            usv.energy = std::stod(match[7].str());
            usv.max_energy = std::stod(match[8].str());
            usv.base_energy_consumption = std::stod(match[9].str());
            usv.payload_factor = std::stod(match[10].str());
            usvs.push_back(usv);
        }
        // 尝试旧格式（兼容）
        else if (std::regex_search(line, match, usv_old_regex)) {
            USV usv;
            usv.id = std::stoi(match[1].str());
            usv.x = std::stoi(match[2].str());
            usv.y = std::stoi(match[3].str());
            usv.max_speed = std::stod(match[4].str());
            usv.capacity = std::stoi(match[5].str());
            // 设置默认值
            usv.volume_capacity = usv.capacity * 2;
            usv.energy = 8000.0;
            usv.max_energy = 10000.0;
            usv.base_energy_consumption = 0.5;
            usv.payload_factor = 0.1;
            usvs.push_back(usv);
        }
    }
    return usvs;
}

std::unordered_map<int, Port> parsePortFile(const std::string& filepath) {
    std::unordered_map<int, Port> ports;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开 Port 文件: " << filepath << std::endl;
        return ports;
    }

    std::string line;
    std::regex port_regex(R"(Port\s+(\d+):\s*\((\d+),\s*(\d+)\))");
    std::smatch match;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (std::regex_search(line, match, port_regex)) {
            Port port;
            port.id = std::stoi(match[1].str());
            port.x = std::stoi(match[2].str());
            port.y = std::stoi(match[3].str());
            ports[port.id] = port;
        }
    }
    return ports;
}

std::vector<GasStation> parseGasStationFile(const std::string& filepath) {
    std::vector<GasStation> stations;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "警告: 无法打开 GasStation 文件: " << filepath << std::endl;
        return stations;
    }

    std::string line;
    std::regex station_regex(R"(Station\s+(\d+):\s*\((\d+)\.00,\s*(\d+)\.00\))");
    std::smatch match;
    int id = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (std::regex_search(line, match, station_regex)) {
            GasStation station;
            station.id = id++;
            station.x = std::stoi(match[2].str());
            station.y = std::stoi(match[3].str());
            station.node_id = -1;
            station.refuel_rate = 100.0;
            stations.push_back(station);
        }
    }
    return stations;
}

std::vector<TransportTask> parseTaskFile(const std::string& filepath) {
    std::vector<TransportTask> tasks;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open Task file: " << filepath << std::endl;
        return tasks;
    }
    std::string line;
    std::regex task_regex(R"(Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))");
    std::smatch match;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (std::regex_search(line, match, task_regex)) {
            TransportTask t;
            t.id = std::stoi(match[1].str());
            t.pickup_x = std::stoi(match[2].str());
            t.pickup_y = std::stoi(match[3].str());
            t.delivery_x = std::stoi(match[4].str());
            t.delivery_y = std::stoi(match[5].str());
            tasks.push_back(t);
        }
    }
    return tasks;
}
