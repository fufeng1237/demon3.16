#!/usr/bin/env python3
"""
交互式任务选择器 — 在地图上点击选择装货/卸货点, 保存到 tasks.txt

操作:
  左键点击 — 放置装货点 (绿色圆圈), 再点击放置卸货点 (蓝色叉), 完成一个任务
  右键点击 — 撤销上一个点
  按 's' — 保存到 tasks.txt
  按 'q' — 退出
  按 'u' — 撤销上一个任务
"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

# 加载路网和地图
rn = load_road_network(f'{BASE}/../output/road_network.json', f'{BASE}/../config/ports.yaml')
img = np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
if img.ndim == 3: img = img[:,:,0]
h_px, w_px = img.shape; ps = 2.0
bg = np.flipud(img); world_w, world_h = w_px * ps, h_px * ps
water = bg > 127; wr, wc = np.where(water); m = 50

tasks = []           # [(pickup_x, pickup_y, delivery_x, delivery_y), ...]
current_pickup = None # 当前正在放置的装货点
markers = []          # matplotlib 标记

fig, ax = plt.subplots(figsize=(20, 9))
ax.imshow(bg, extent=[0, world_w, 0, world_h], origin='lower', cmap='gray')
for e in rn.edges:
    n1, n2 = rn.nodes[e.from_id], rn.nodes[e.to_id]
    ax.plot([n1.x, n2.x], [world_h-n1.y, world_h-n2.y], 'cyan', lw=0.4, alpha=0.3, zorder=1)
for n in rn.nodes.values():
    if n.is_port:
        ax.scatter(n.x, world_h-n.y, c='red', s=80, marker='s', edgecolors='white', lw=1.5, zorder=3)
        ax.annotate(n.port_name, (n.x, world_h-n.y+25), fontsize=7, color='white', ha='center', zorder=5)

ax.set_xlim(max(0, wc.min()-m)*ps, min(w_px, wc.max()+m)*ps)
ax.set_ylim(max(0, wr.min()-m)*ps, min(h_px, wr.max()+m)*ps)
ax.set_aspect('equal')
ax.set_title('Interactive Task Picker\n'
             'Left-click: pickup → delivery | Right-click: undo point | '
             f's: save | q: quit | Tasks: 0',
             fontsize=11, color='white')
fig.patch.set_facecolor('#111111')
ax.set_facecolor('#111111')

info_text = ax.text(0.02, 0.02, 'Click to place Pickup point', transform=ax.transAxes,
                    fontsize=10, color='lime', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))


def update_display():
    ax.set_title(f'Interactive Task Picker\n'
                 f'Left-click: pickup → delivery | Right-click: undo point | '
                 f's: save | q: quit | Tasks: {len(tasks)}',
                 fontsize=11, color='white')
    status = f'Tasks: {len(tasks)}  |  '
    if current_pickup:
        px, py = current_pickup
        status += f'Pickup set at ({px:.0f}, {world_h-py:.0f})m — now click Delivery point'
    else:
        status += 'Click to place Pickup point'
    info_text.set_text(status)
    fig.canvas.draw()


def snap_to_water(wx, wy):
    """吸附到最近水域像素"""
    px = int(wx / ps); py = int(wy / ps)
    best = (px, py); best_d = 9999
    for r in range(max(0, py-20), min(h_px, py+21)):
        for c in range(max(0, px-20), min(w_px, px+21)):
            if water[r, c]:
                d = (c-px)**2 + (r-py)**2
                if d < best_d: best_d = d; best = (c, r)
    return best[0] * ps, best[1] * ps


def onclick(event):
    global current_pickup
    if event.inaxes != ax: return
    if event.button == 1:  # 左键
        wx, wy = event.xdata, event.ydata
        wx, wy = snap_to_water(wx, wy)
        world_y = world_h - wy  # B版坐标转换

        if current_pickup is None:
            # 放置装货点
            current_pickup = (wx, world_y)
            m = ax.scatter(wx, wy, c='lime', s=120, marker='o', edgecolors='white', lw=2, zorder=8)
            markers.append(m)
            ax.annotate(f'P{len(tasks)}', (wx, wy+20), fontsize=8, color='lime',
                       fontweight='bold', ha='center', zorder=9)
        else:
            # 放置卸货点 → 完成任务
            pickup_wx, pickup_wy = current_pickup
            tasks.append((pickup_wx, pickup_wy, wx, world_y))
            m = ax.scatter(wx, wy, c='#ff4444', s=120, marker='X', edgecolors='white', lw=2, zorder=8)
            markers.append(m)
            ax.annotate(f'D{len(tasks)-1}', (wx, wy+20), fontsize=8, color='#ff4444',
                       fontweight='bold', ha='center', zorder=9)
            # 连线
            pickup_display_y = world_h - pickup_wy
            ax.plot([pickup_wx, wx], [pickup_display_y, wy], 'yellow', lw=1.5, alpha=0.6, zorder=7)
            ax.annotate(f'T{len(tasks)-1}', ((pickup_wx+wx)/2, (pickup_display_y+wy)/2),
                       fontsize=7, color='yellow', ha='center', zorder=9,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
            current_pickup = None

    elif event.button == 3:  # 右键 — 撤销
        if current_pickup:
            current_pickup = None
            if markers: markers.pop().remove()
            if markers: markers.pop().remove()

    update_display()


def onkey(event):
    if event.key == 's':
        output = f'{BASE}/../config/tasks.txt'
        with open(output, 'w') as f:
            f.write(f"# Tasks — pickup_x pickup_y delivery_x delivery_y (B-version world coords)\n")
            f.write(f"# Format: Task ID: (pickup_x, pickup_y) -> (delivery_x, delivery_y)\n\n")
            for i, (px, py, dx, dy) in enumerate(tasks):
                f.write(f"Task {i}: pickup=({px:.0f},{py:.0f}) delivery=({dx:.0f},{dy:.0f})\n")
        print(f'\nSaved {len(tasks)} tasks to: {output}')
        info_text.set_text(f'Saved {len(tasks)} tasks to {output.split("/")[-1]}!')
        fig.canvas.draw()

    elif event.key == 'q':
        print(f'\nExiting. Total tasks: {len(tasks)}')
        plt.close()

    elif event.key == 'u':
        if tasks:
            t = tasks.pop()
            print(f'Undo: Task {len(tasks)} removed ({t[0]:.0f},{t[1]:.0f}→{t[2]:.0f},{t[3]:.0f})')
            for _ in range(4):  # 移除标记和文字
                if markers: markers.pop().remove()
            update_display()


fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

plt.tight_layout()
print("Interactive Task Picker")
print("  Left-click: pickup → delivery  |  Right-click: undo point")
print("  s: save to tasks.txt  |  u: undo last task  |  q: quit")
plt.show()
