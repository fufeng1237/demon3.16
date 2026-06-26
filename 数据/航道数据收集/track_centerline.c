#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

// 定义点结构体
typedef struct {
    double lon;
    double lat;
} Point;

// 定义轨迹结构体
typedef struct {
    Point* points;
    int count;
} Track;

// 读取CSV文件（假设Excel已转换为CSV）
Track* read_csv_file(const char* file_path) {
    FILE* fp = fopen(file_path, "r");
    if (!fp) {      
        printf("无法打开文件: %s\n", file_path);
        return NULL;
    }

    // 计算文件中的行数
    int line_count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        line_count++;
    }
    rewind(fp);

    // 分配内存
    Track* track = (Track*)malloc(sizeof(Track));
    track->points = (Point*)malloc(sizeof(Point) * line_count);
    track->count = 0;

    // 读取数据
    char* token;
    while (fgets(buffer, sizeof(buffer), fp)) {
        token = strtok(buffer, ",");
        if (token) {
            // 假设第一列为经度，第二列为纬度
            track->points[track->count].lon = atof(token);
            token = strtok(NULL, ",");
            if (token) {
                track->points[track->count].lat = atof(token);
                track->count++;
            }
        }
    }

    fclose(fp);
    return track;
}

// 三次样条插值结构体
typedef struct {
    double* x;
    double* y;
    double* y2;
    int n;
} Spline;

// 三次样条插值初始化
Spline* spline_init(double* x, double* y, int n) {
    Spline* spline = (Spline*)malloc(sizeof(Spline));
    spline->x = x;
    spline->y = y;
    spline->n = n;
    spline->y2 = (double*)malloc(sizeof(double) * n);

    // 计算二阶导数
    double* u = (double*)malloc(sizeof(double) * (n-1));
    spline->y2[0] = 0.0;
    u[0] = 0.0;

    for (int i = 1; i < n-1; i++) {
        double p = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);
        double q = p * spline->y2[i-1] + 2.0;
        spline->y2[i] = (p - 1.0) / q;
        u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1]);
        u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - p * u[i-1]) / q;
    }

    spline->y2[n-1] = 0.0;
    for (int i = n-2; i >= 0; i--) {
        spline->y2[i] = spline->y2[i] * spline->y2[i+1] + u[i];
    }

    free(u);
    return spline;
}

// 三次样条插值计算
double spline_eval(Spline* spline, double x) {
    int i, j, k;
    double h, b, a;

    // 二分查找
    i = 0;
    j = spline->n - 1;
    while (j - i > 1) {
        k = (i + j) / 2;
        if (spline->x[k] > x) {
            j = k;
        } else {
            i = k;
        }
    }

    h = spline->x[j] - spline->x[i];
    a = (spline->x[j] - x) / h;
    b = (x - spline->x[i]) / h;

    return a * spline->y[i] + b * spline->y[j] + 
           ((a*a*a - a) * spline->y2[i] + (b*b*b - b) * spline->y2[j]) * h*h / 6.0;
}

// 计算轨迹中心线
Track* calculate_centerline(Track* track) {
    if (track->count < 2) {
        return track;
    }

    // 创建参数t
    int n = track->count;
    double* t = (double*)malloc(sizeof(double) * n);
    double* lons = (double*)malloc(sizeof(double) * n);
    double* lats = (double*)malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        t[i] = (double)i / (n-1);
        lons[i] = track->points[i].lon;
        lats[i] = track->points[i].lat;
    }

    // 初始化样条插值
    Spline* spline_lon = spline_init(t, lons, n);
    Spline* spline_lat = spline_init(t, lats, n);

    // 生成新的点
    int new_count = 100;
    Track* centerline = (Track*)malloc(sizeof(Track));
    centerline->points = (Point*)malloc(sizeof(Point) * new_count);
    centerline->count = new_count;

    for (int i = 0; i < new_count; i++) {
        double t_new = (double)i / (new_count-1);
        centerline->points[i].lon = spline_eval(spline_lon, t_new);
        centerline->points[i].lat = spline_eval(spline_lat, t_new);
    }

    // 释放内存
    free(t);
    free(lons);
    free(lats);
    free(spline_lon->y2);
    free(spline_lon);
    free(spline_lat->y2);
    free(spline_lat);

    return centerline;
}

// 将轨迹分成上下两个通道
void split_into_channels(Track* centerline, Track* channel1, Track* channel2) {
    if (centerline->count < 2) {
        return;
    }

    // 计算轨迹方向
    double sum_dx = 0.0, sum_dy = 0.0;
    for (int i = 0; i < centerline->count - 1; i++) {
        sum_dx += centerline->points[i+1].lon - centerline->points[i].lon;
        sum_dy += centerline->points[i+1].lat - centerline->points[i].lat;
    }

    double mean_angle = atan2(sum_dy, sum_dx);
    double perpendicular_angle1 = mean_angle + M_PI / 2.0;
    double perpendicular_angle2 = mean_angle - M_PI / 2.0;

    // 偏移距离（约10米）
    double offset = 0.0001;

    // 分配内存
    channel1->points = (Point*)malloc(sizeof(Point) * centerline->count);
    channel1->count = centerline->count;
    channel2->points = (Point*)malloc(sizeof(Point) * centerline->count);
    channel2->count = centerline->count;

    // 计算通道点
    for (int i = 0; i < centerline->count; i++) {
        channel1->points[i].lon = centerline->points[i].lon + offset * cos(perpendicular_angle1);
        channel1->points[i].lat = centerline->points[i].lat + offset * sin(perpendicular_angle1);
        channel2->points[i].lon = centerline->points[i].lon + offset * cos(perpendicular_angle2);
        channel2->points[i].lat = centerline->points[i].lat + offset * sin(perpendicular_angle2);
    }
}

// 保存轨迹数据到CSV文件
void save_track_to_csv(Track* track, const char* file_path) {
    FILE* fp = fopen(file_path, "w");
    if (!fp) {
        printf("无法创建文件: %s\n", file_path);
        return;
    }

    fprintf(fp, "经度,纬度\n");
    for (int i = 0; i < track->count; i++) {
        fprintf(fp, "%lf,%lf\n", track->points[i].lon, track->points[i].lat);
    }

    fclose(fp);
}

// 处理单个文件
void process_file(const char* file_path) {
    printf("处理文件: %s\n", file_path);

    // 读取CSV文件（假设Excel已转换为CSV）
    Track* track = read_csv_file(file_path);
    if (!track) {
        return;
    }

    // 计算中心线
    Track* centerline = calculate_centerline(track);

    // 分成上下通道
    Track channel1, channel2;
    split_into_channels(centerline, &channel1, &channel2);

    // 生成输出文件名
    char output_dir[] = "output";
    CreateDirectoryA(output_dir, NULL);

    char base_name[256];
    char* ext = strrchr(file_path, '.');
    if (ext) {
        strncpy(base_name, file_path, ext - file_path);
        base_name[ext - file_path] = '\0';
    } else {
        strcpy(base_name, file_path);
    }

    char centerline_file[512];
    char channel1_file[512];
    char channel2_file[512];

    sprintf(centerline_file, "%s\\%s_centerline.csv", output_dir, strrchr(base_name, '\\') ? strrchr(base_name, '\\') + 1 : base_name);
    sprintf(channel1_file, "%s\\%s_channel1.csv", output_dir, strrchr(base_name, '\\') ? strrchr(base_name, '\\') + 1 : base_name);
    sprintf(channel2_file, "%s\\%s_channel2.csv", output_dir, strrchr(base_name, '\\') ? strrchr(base_name, '\\') + 1 : base_name);

    // 保存数据
    save_track_to_csv(centerline, centerline_file);
    save_track_to_csv(&channel1, channel1_file);
    save_track_to_csv(&channel2, channel2_file);

    // 释放内存
    free(track->points);
    free(track);
    free(centerline->points);
    free(centerline);
    free(channel1.points);
    free(channel2.points);

    printf("文件处理完成: %s\n", file_path);
}

// 遍历目录处理所有文件
void process_directory(const char* directory) {
    char search_path[512];
    sprintf(search_path, "%s\\*.csv", directory);

    WIN32_FIND_DATAA find_data;
    HANDLE hFind = FindFirstFileA(search_path, &find_data);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char file_path[512];
                sprintf(file_path, "%s\\%s", directory, find_data.cFileName);
                process_file(file_path);
            }
        } while (FindNextFileA(hFind, &find_data));
        FindClose(hFind);
    }

    // 处理子目录
    sprintf(search_path, "%s\\*", directory);
    hFind = FindFirstFileA(search_path, &find_data);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if ((find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && 
                strcmp(find_data.cFileName, ".") != 0 && 
                strcmp(find_data.cFileName, "..") != 0) {
                char subdir_path[512];
                sprintf(subdir_path, "%s\\%s", directory, find_data.cFileName);
                process_directory(subdir_path);
            }
        } while (FindNextFileA(hFind, &find_data));
        FindClose(hFind);
    }
}

int main() {
    // 处理所有CSV文件
    const char* base_dir = "d:\\工作文件\\小船数据\\新建文件夹\\航道数据收集";
    process_directory(base_dir);
    printf("所有文件处理完成！\n");
    return 0;
}