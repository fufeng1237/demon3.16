#!/bin/bash
# 一键启动 Xvfb + VNC + map + planner

# 1. 启动虚拟显示器
pkill Xvfb 2>/dev/null
Xvfb :99 -screen 0 1920x1080x24 &>/tmp/xvfb.log &
sleep 1

# 2. 启动 VNC
pkill x11vnc 2>/dev/null
x11vnc -display :99 -forever -nopw -bg -o /tmp/x11vnc.log
sleep 1

# 3. 启动 noVNC (Web 版 VNC 客户端)
pkill -f websockify 2>/dev/null
websockify --web=/usr/share/novnc 6080 localhost:5900 &>/tmp/websockify.log &
sleep 2

echo "=== VNC 服务已就绪 ==="
echo "在 VS Code 中转发端口 6080 并在浏览器打开"
echo ""

# 4. 启动 ROS
source /opt/ros/noetic/setup.bash
source /root/demon3.16/devel/setup.bash
export DISPLAY=:99

# 5. 启动 map.launch
roslaunch waterway_map map.launch &
sleep 5

# 6. 启动 planner.launch
roslaunch waterway_map planner.launch &

echo "=== 所有服务已启动 ==="
wait
