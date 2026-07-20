#!/usr/bin/env python3
"""路网可视化 — B 版: flip bg + origin='lower' + extent=[0,W,0,H] + fy(node.y)"""

import numpy as np, json, os
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_road_network_on_map(network, map_png, output, ds=1, dpi=150):
    """B version: flip bg, origin='lower', node y = world_h - world_y"""
    img = np.array(Image.open(map_png))
    if img.ndim==3: img = img[:,:,0]
    h, w = img.shape; ps = 2.0; wh = h*ps; ww = w*ps

    bg = np.flipud(img)  # B flip
    water = bg>127; wr,wc = np.where(water); m=30
    r0=max(0,wr.min()-m)*ps; r1=min(h,wr.max()+m)*ps
    c0=max(0,wc.min()-m)*ps; c1=min(w,wc.max()+m)*ps

    fig, ax = plt.subplots(figsize=(20,7))
    ax.imshow(bg, extent=[0,ww,0,wh], origin='lower', cmap='gray')

    for e in network.edges:
        n1, n2 = network.nodes[e.from_id], network.nodes[e.to_id]
        ax.plot([n1.x,n2.x], [wh-n1.y,wh-n2.y], '#00ffdd', lw=3, alpha=0.8, zorder=3)

    for n in network.nodes.values():
        ny = wh - n.y
        if n.is_port:
            ax.scatter(n.x, ny, c='red', s=36, marker='s', edgecolors='white', lw=0.8, zorder=6)
            if n.port_name:
                ax.annotate(n.port_name, (n.x, ny), xytext=(4,4), textcoords='offset points',
                            fontsize=6, color='white', fontweight='bold', zorder=8)
        elif n.is_gas_station:
            ax.scatter(n.x, ny, c='lime', s=120, marker='^', edgecolors='white', lw=1.5, zorder=6)
            ax.annotate(n.port_name, (n.x, ny), xytext=(5,5), textcoords='offset points',
                        fontsize=8, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.85), zorder=8)
        elif n.is_task_anchor:
            ax.scatter(n.x, ny, c='#ffb000', s=8, alpha=0.65, zorder=5)
        elif hasattr(n, 'node_type') and str(n.node_type)!='node':
            pass
        else:
            ax.scatter(n.x, ny, c='yellow', s=15, alpha=0.7, zorder=4)

    ax.set_xlim(c0,c1); ax.set_ylim(r0,r1); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    print(f"  Saved: {output}")
