%% 修正版：从文件读取真实轨迹数据并精确匹配
clear; clc; close all;

fprintf('=== 真实轨迹数据与栅格地图精确匹配 ===\n');

%% 1. 加载栅格地图
binaryMap = imread('binary_map_manually_cleaned1.png');
if size(binaryMap, 3) == 3
    binaryMap = rgb2gray(binaryMap) > 128;
end
binaryMap = logical(binaryMap);
[rows, cols] = size(binaryMap);
fprintf('栅格地图尺寸: %d x %d 像素\n', cols, rows);

% 创建RGB版本用于显示彩色轨迹
rgbMap = uint8(cat(3, binaryMap*255, binaryMap*255, binaryMap*255));

%% 2. 关键步骤：手动建立栅格地图的地理参考
% 从您的图片和轨迹数据可以看出，栅格地图覆盖了特定的区域
% 根据您提供的参数进行调整：
map_lon_min = 120.414716;   % 栅格地图最小经度
map_lon_max = 120.644716;   % 栅格地图最大经度
map_lat_min = 31.980283;   % 栅格地图最小纬度
map_lat_max = 32.081950;   % 栅格地图最大纬度

fprintf('估计的栅格地图范围:\n');
fprintf('  经度: %.6f 到 %.6f\n', map_lon_min, map_lon_max);
fprintf('  纬度: %.6f 到 %.6f\n', map_lat_min, map_lat_max);

%% 3. 创建坐标转换函数
% 经纬度 -> 像素坐标
lon_to_pixel = @(lon) 1 + (lon - map_lon_min)/(map_lon_max - map_lon_min) * (cols - 1);
lat_to_pixel = @(lat) 1 + (map_lat_max - lat)/(map_lat_max - map_lat_min) * (rows - 1);

% 像素坐标 -> 经纬度
pixel_to_lon = @(x) map_lon_min + (x-1)/(cols-1) * (map_lon_max - map_lon_min);
pixel_to_lat = @(y) map_lat_max - (y-1)/(rows-1) * (map_lat_max - map_lat_min);

%% 4. 从文件读取真实轨迹数据
fprintf('\n=== 从文件读取真实轨迹数据 ===\n');

% 设置轨迹文件夹路径（请根据您的实际情况修改）
left_folder = 'D:\工作文件\小船数据\航道数据收集\0-80';    % 向左走船舶数据文件夹
right_folder = 'D:\工作文件\小船数据\向右';              % 向右走船舶数据文件夹

% 检查文件夹是否存在
if ~exist(left_folder, 'dir')
    fprintf('警告: 向左走船舶文件夹不存在: %s\n', left_folder);
    left_data = [];
else
    fprintf('读取向左走船舶数据...\n');
    left_data = process_ship_folder(left_folder, 'left', lon_to_pixel, lat_to_pixel);
end

if ~exist(right_folder, 'dir')
    fprintf('警告: 向右走船舶文件夹不存在: %s\n', right_folder);
    right_data = [];
else
    fprintf('读取向右走船舶数据...\n');
    right_data = process_ship_folder(right_folder, 'right', lon_to_pixel, lat_to_pixel);
end

fprintf('数据读取完成:\n');
fprintf('  向左走: %d 条轨迹\n', length(left_data));
fprintf('  向右走: %d 条轨迹\n', length(right_data));

%% 5. 可视化匹配结果
figure('Position', [100, 100, 1400, 600], 'Name', '真实轨迹与栅格地图精确匹配');

% 5.1 左侧：栅格地图上的轨迹
subplot(1, 2, 1);
imshow(rgbMap);
hold on;

% 绘制向左走轨迹（红色）
if ~isempty(left_data)
    for i = 1:min(20, length(left_data))
        if isfield(left_data(i), 'pixel_x') && ~isempty(left_data(i).pixel_x)
            plot(left_data(i).pixel_x, left_data(i).pixel_y, ...
                'r-', 'LineWidth', 1.5);
            plot(left_data(i).pixel_x(1), left_data(i).pixel_y(1), ...
                'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
            plot(left_data(i).pixel_x(end), left_data(i).pixel_y(end), ...
                'rs', 'MarkerSize', 6, 'MarkerFaceColor', [0.8, 0, 0]);
        end
    end
end

% 绘制向右走轨迹（蓝色）
if ~isempty(right_data)
    for i = 1:min(20, length(right_data))
        if isfield(right_data(i), 'pixel_x') && ~isempty(right_data(i).pixel_x)
            plot(right_data(i).pixel_x, right_data(i).pixel_y, ...
                'b-', 'LineWidth', 1.5);
            plot(right_data(i).pixel_x(1), right_data(i).pixel_y(1), ...
                'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
            plot(right_data(i).pixel_x(end), right_data(i).pixel_y(end), ...
                'bs', 'MarkerSize', 6, 'MarkerFaceColor', [0, 0, 0.8]);
        end
    end
end

title('栅格地图上的真实船舶轨迹', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('像素列', 'FontSize', 12);
ylabel('像素行', 'FontSize', 12);

% 添加图例
legend_items = {};
if ~isempty(left_data)
    h_left = plot(NaN, NaN, 'r-', 'LineWidth', 2);
    legend_items{end+1} = sprintf('向左走 (%d艘)', length(left_data));
end
if ~isempty(right_data)
    h_right = plot(NaN, NaN, 'b-', 'LineWidth', 2);
    legend_items{end+1} = sprintf('向右走 (%d艘)', length(right_data));
end

if ~isempty(legend_items)
    if ~isempty(left_data) && ~isempty(right_data)
        legend([h_left, h_right], legend_items, 'Location', 'best');
    elseif ~isempty(left_data)
        legend(h_left, legend_items{1}, 'Location', 'best');
    elseif ~isempty(right_data)
        legend(h_right, legend_items{1}, 'Location', 'best');
    end
end

% 添加角点标注
text(10, 10, sprintf('左上: (%.3f, %.3f)', map_lon_min, map_lat_max), ...
    'Color', 'blue', 'FontSize', 9, 'BackgroundColor', 'white', ...
    'VerticalAlignment', 'top');
text(cols-10, 10, sprintf('右上: (%.3f, %.3f)', map_lon_max, map_lat_max), ...
    'Color', 'blue', 'FontSize', 9, 'BackgroundColor', 'white', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
text(cols-10, rows-10, sprintf('右下: (%.3f, %.3f)', map_lon_max, map_lat_min), ...
    'Color', 'blue', 'FontSize', 9, 'BackgroundColor', 'white', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
text(10, rows-10, sprintf('左下: (%.3f, %.3f)', map_lon_min, map_lat_min), ...
    'Color', 'blue', 'FontSize', 9, 'BackgroundColor', 'white', ...
    'VerticalAlignment', 'bottom');

hold off;

% 5.2 右侧：经纬度坐标系中的轨迹
subplot(1, 2, 2);
hold on;
grid on;

% 设置坐标轴范围
xlim([120.0, 120.9]);
ylim([31.9, 32.1]);

% 绘制经纬度网格
for lon = 120.0:0.1:120.9
    plot([lon, lon], [31.9, 32.1], 'k:', 'LineWidth', 0.5);
end
for lat = 31.9:0.05:32.1
    plot([120.0, 120.9], [lat, lat], 'k:', 'LineWidth', 0.5);
end

% 绘制向左走轨迹（经纬度坐标）
if ~isempty(left_data)
    for i = 1:min(10, length(left_data))
        if isfield(left_data(i), 'longitude') && ~isempty(left_data(i).longitude)
            plot(left_data(i).longitude, left_data(i).latitude, ...
                'r-', 'LineWidth', 1.0);
        end
    end
end

% 绘制向右走轨迹（经纬度坐标）
if ~isempty(right_data)
    for i = 1:min(10, length(right_data))
        if isfield(right_data(i), 'longitude') && ~isempty(right_data(i).longitude)
            plot(right_data(i).longitude, right_data(i).latitude, ...
                'b-', 'LineWidth', 1.0);
        end
    end
end

% 标注栅格地图范围
rectangle('Position', [map_lon_min, map_lat_min, ...
    map_lon_max-map_lon_min, map_lat_max-map_lat_min], ...
    'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '--');
text(map_lon_min, map_lat_max+0.005, '栅格地图范围', ...
    'Color', 'g', 'FontSize', 10, 'FontWeight', 'bold');

xlabel('经度', 'FontSize', 12);
ylabel('纬度', 'FontSize', 12);
title('经纬度坐标系中的真实轨迹', 'FontSize', 14, 'FontWeight', 'bold');

% 添加比例尺
text(120.05, 31.92, '经度: 120.0-120.9', 'FontSize', 10, 'BackgroundColor', 'white');
text(120.05, 31.905, '纬度: 31.9-32.1', 'FontSize', 10, 'BackgroundColor', 'white');

hold off;

%% 6. 分析轨迹统计信息
fprintf('\n=== 轨迹数据统计 ===\n');

total_points = 0;
if ~isempty(left_data)
    for i = 1:length(left_data)
        if isfield(left_data(i), 'longitude')
            total_points = total_points + length(left_data(i).longitude);
        end
    end
    fprintf('向左走轨迹总点数: %d\n', total_points);
end

total_points = 0;
if ~isempty(right_data)
    for i = 1:length(right_data)
        if isfield(right_data(i), 'longitude')
            total_points = total_points + length(right_data(i).longitude);
        end
    end
    fprintf('向右走轨迹总点数: %d\n', total_points);
end

%% 7. 保存结果
saveas(gcf, 'real_trajectory_map_match.png');
fprintf('\n结果已保存: real_trajectory_map_match.png\n');

%% 8. 函数：处理船舶文件夹
function ship_data = process_ship_folder(folder_path, direction, lon_to_pixel, lat_to_pixel)
    % 获取所有Excel文件
    xls_files = dir(fullfile(folder_path, '*.xls'));
    xlsx_files = dir(fullfile(folder_path, '*.xlsx'));
    all_files = [xls_files; xlsx_files];
    
    if isempty(all_files)
        fprintf('  文件夹中没有Excel文件: %s\n', folder_path);
        ship_data = [];
        return;
    end
    
    fprintf('  在 %s 中找到 %d 个文件\n', folder_path, length(all_files));
    
    ship_data = struct('longitude', {}, 'latitude', {}, 'pixel_x', {}, 'pixel_y', {}, 'mmsi', {});
    valid_count = 0;
    
    for i = 1:length(all_files)
        file_path = fullfile(folder_path, all_files(i).name);
        
        try
            % 读取Excel文件
            [longitude, latitude, mmsi] = read_ship_excel(file_path);
            
            if ~isempty(longitude) && length(longitude) > 1
                valid_count = valid_count + 1;
                
                % 将经纬度转换为像素坐标
                pixel_x = lon_to_pixel(longitude);
                pixel_y = lat_to_pixel(latitude);
                
                % 存储数据
                ship_data(valid_count).longitude = longitude;
                ship_data(valid_count).latitude = latitude;
                ship_data(valid_count).pixel_x = pixel_x;
                ship_data(valid_count).pixel_y = pixel_y;
                ship_data(valid_count).mmsi = mmsi;
                ship_data(valid_count).direction = direction;
                ship_data(valid_count).filename = all_files(i).name;
                
                fprintf('    已处理: %s (%d个点)\n', all_files(i).name, length(longitude));
            end
            
        catch ME
            fprintf('    处理文件失败: %s (%s)\n', all_files(i).name, ME.message);
        end
    end
    
    fprintf('  有效文件: %d/%d\n', valid_count, length(all_files));
end

%% 9. 函数：读取船舶Excel文件
function [longitude, latitude, mmsi] = read_ship_excel(file_path)
    longitude = [];
    latitude = [];
    mmsi = NaN;
    
    try
        % 读取Excel文件
        [num_data, txt_data, raw_data] = xlsread(file_path);
        
        if isempty(raw_data) || size(raw_data, 1) < 2
            fprintf('      文件为空或行数不足\n');
            return;
        end
        
        % 查找经度、纬度、MMSI列
        lon_col = 0;
        lat_col = 0;
        mmsi_col = 0;
        
        % 尝试查找表头
        for col = 1:min(10, size(raw_data, 2))
            if iscell(raw_data{1, col}) || ischar(raw_data{1, col}) || isstring(raw_data{1, col})
                header = lower(strtrim(char(raw_data{1, col})));
                if contains(header, '经度') || contains(header, 'lon')
                    lon_col = col;
                elseif contains(header, '纬度') || contains(header, 'lat')
                    lat_col = col;
                elseif contains(header, 'mmsi')
                    mmsi_col = col;
                end
            end
        end
        
        % 如果没有找到表头，尝试默认列位置
        if lon_col == 0 && size(raw_data, 2) >= 2
            lon_col = 2;
        end
        if lat_col == 0 && size(raw_data, 2) >= 3
            lat_col = 3;
        end
        if mmsi_col == 0 && size(raw_data, 2) >= 1
            mmsi_col = 1;
        end
        
        % 提取数据
        for row = 2:size(raw_data, 1)
            % 提取经度
            if lon_col > 0 && lon_col <= size(raw_data, 2)
                lon_val = raw_data{row, lon_col};
                if isnumeric(lon_val) && ~isnan(lon_val)
                    longitude(end+1) = lon_val;
                elseif ischar(lon_val) || isstring(lon_val)
                    lon_num = str2double(lon_val);
                    if ~isnan(lon_num)
                        longitude(end+1) = lon_num;
                    end
                end
            end
            
            % 提取纬度
            if lat_col > 0 && lat_col <= size(raw_data, 2)
                lat_val = raw_data{row, lat_col};
                if isnumeric(lat_val) && ~isnan(lat_val)
                    latitude(end+1) = lat_val;
                elseif ischar(lat_val) || isstring(lat_val)
                    lat_num = str2double(lat_val);
                    if ~isnan(lat_num)
                        latitude(end+1) = lat_num;
                    end
                end
            end
        end
        
        % 提取MMSI
        if mmsi_col > 0 && mmsi_col <= size(raw_data, 2) && size(raw_data, 1) >= 2
            mmsi_val = raw_data{2, mmsi_col};
            if isnumeric(mmsi_val)
                mmsi = mmsi_val;
            elseif ischar(mmsi_val) || isstring(mmsi_val)
                mmsi = str2double(mmsi_val);
            end
        end
        
        % 如果从文件无法获取MMSI，尝试从文件名提取
        if isnan(mmsi)
            [~, filename] = fileparts(file_path);
            filename_parts = strsplit(filename, '_');
            if length(filename_parts) >= 1
                mmsi = str2double(filename_parts{1});
            end
        end
        
    catch ME
        fprintf('      读取文件出错: %s\n', ME.message);
    end
end