import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

# 读取Excel文件
def read_excel_file(file_path):
    try:
        # 尝试不同的sheet名称
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
        except:
            df = pd.read_excel(file_path)
        
        # 识别经纬度列
        lat_col = None
        lon_col = None
        
        # 检查列名
        for col in df.columns:
            col_lower = str(col).lower()
            if 'lat' in col_lower or '纬度' in col_lower:
                lat_col = col
            elif 'lon' in col_lower or '经度' in col_lower:
                lon_col = col
        
        # 如果没有找到，尝试根据数据特征识别
        if lat_col is None or lon_col is None:
            # 假设前几列中包含经纬度
            for i, col in enumerate(df.columns[:5]):
                data = df[col].dropna()
                if len(data) > 0:
                    # 纬度范围通常在-90到90之间
                    if (data.min() >= -90) and (data.max() <= 90):
                        lat_col = col
                    # 经度范围通常在-180到180之间
                    elif (data.min() >= -180) and (data.max() <= 180):
                        lon_col = col
        
        if lat_col is None or lon_col is None:
            print(f"无法识别文件 {file_path} 中的经纬度列")
            return None
        
        # 提取经纬度数据
        lons = df[lon_col].dropna().values
        lats = df[lat_col].dropna().values
        
        return lons, lats
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

# 计算轨迹中心线
def calculate_centerline(lons, lats):
    if len(lons) < 2:
        return lons, lats
    
    # 对轨迹进行插值，使点分布更均匀
    t = np.linspace(0, 1, len(lons))
    t_new = np.linspace(0, 1, max(100, len(lons)))
    
    # 使用样条插值
    lon_interp = interpolate.interp1d(t, lons, kind='cubic')
    lat_interp = interpolate.interp1d(t, lats, kind='cubic')
    
    lons_smooth = lon_interp(t_new)
    lats_smooth = lat_interp(t_new)
    
    return lons_smooth, lats_smooth

# 将轨迹分成上下两个通道
def split_into_channels(lons, lats):
    # 计算轨迹的方向
    dx = np.diff(lons)
    dy = np.diff(lats)
    angles = np.arctan2(dy, dx)
    
    # 计算平均方向
    mean_angle = np.mean(angles)
    
    # 计算垂直方向（向左和向右）
    perpendicular_angle1 = mean_angle + np.pi/2
    perpendicular_angle2 = mean_angle - np.pi/2
    
    # 计算偏移距离（可根据实际情况调整）
    offset = 0.0001  # 约10米
    
    # 计算上下通道的点
    lons_channel1 = lons + offset * np.cos(perpendicular_angle1)
    lats_channel1 = lats + offset * np.sin(perpendicular_angle1)
    
    lons_channel2 = lons + offset * np.cos(perpendicular_angle2)
    lats_channel2 = lats + offset * np.sin(perpendicular_angle2)
    
    return (lons_channel1, lats_channel1), (lons_channel2, lats_channel2)

# 可视化轨迹和中心线
def visualize_track(lons, lats, centerline, channel1, channel2, file_name):
    plt.figure(figsize=(12, 8))
    
    # 绘制原始轨迹
    plt.plot(lons, lats, 'b-', alpha=0.5, label='原始轨迹')
    
    # 绘制中心线
    plt.plot(centerline[0], centerline[1], 'r-', linewidth=2, label='轨迹中心线')
    
    # 绘制上下通道
    plt.plot(channel1[0], channel1[1], 'g-', linewidth=1.5, label='上通道')
    plt.plot(channel2[0], channel2[1], 'y-', linewidth=1.5, label='下通道')
    
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title(f'轨迹分析 - {file_name}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 保存图像
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_track.png'))
    plt.close()

# 处理单个文件
def process_file(file_path):
    print(f"处理文件: {file_path}")
    
    # 读取数据
    data = read_excel_file(file_path)
    if data is None:
        return
    
    lons, lats = data
    
    # 计算中心线
    centerline = calculate_centerline(lons, lats)
    
    # 分成上下通道
    channel1, channel2 = split_into_channels(centerline[0], centerline[1])
    
    # 可视化
    file_name = os.path.basename(file_path)
    visualize_track(lons, lats, centerline, channel1, channel2, file_name)
    
    # 保存中心线和通道数据
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存中心线数据
    centerline_data = pd.DataFrame({
        '经度': centerline[0],
        '纬度': centerline[1]
    })
    centerline_data.to_csv(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_centerline.csv'), index=False)
    
    # 保存通道数据
    channel1_data = pd.DataFrame({
        '经度': channel1[0],
        '纬度': channel1[1]
    })
    channel1_data.to_csv(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_channel1.csv'), index=False)
    
    channel2_data = pd.DataFrame({
        '经度': channel2[0],
        '纬度': channel2[1]
    })
    channel2_data.to_csv(os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_channel2.csv'), index=False)
    
    print(f"文件处理完成: {file_path}")

# 处理目录中的所有文件
def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xls') or file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                process_file(file_path)

if __name__ == '__main__':
    # 处理所有文件
    base_dir = 'd:\\工作文件\\小船数据\\新建文件夹\\航道数据收集'
    process_directory(base_dir)
    print("所有文件处理完成！")