#!/usr/bin/env python3
"""从 config 文件读取船和任务，运行分配"""
import sys,os,re,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from scheduler import Scheduler
from real_time_scheduler import RealTimeScheduler

BASE=os.path.dirname(os.path.abspath(__file__))

# 加载路网
rn=load_road_network(f'{BASE}/../output/road_network.json',f'{BASE}/../config/ports.yaml')
port_ids=sorted([nid for nid,n in rn.nodes.items() if n.is_port])
gas_ids=sorted([nid for nid,n in rn.nodes.items() if n.is_gas_station])
node_names={nid:(n.port_name if n.port_name else f'N{nid}') for nid,n in rn.nodes.items()}

# 找每个船/任务对应的路网节点 (最近邻)
def find_nearest_node(wx,wy):
    best,best_d=None,float('inf')
    for nid,n in rn.nodes.items():
        d=np.sqrt((n.x-wx)**2+(n.y-wy)**2)
        if d<best_d: best_d=d; best=nid
    return best

# ── 读取 ships (usvs.txt, C++像素格式) ──
ships=[]
with open(f'{BASE}/../config/usvs.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        m=re.match(r'USV:\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)',line)
        if m:
            sid=int(m.group(1)); px=int(m.group(2)); py=int(m.group(3))
            sp=float(m.group(4)); cap=int(m.group(5))
            en=float(m.group(7)); max_en=float(m.group(8))
            # C++像素 → 世界坐标 (scale_factor=1.0, 即像素=坐标)
            wx=px*2.0; wy=py*2.0
            nid=find_nearest_node(wx,wy)
            ships.append((sid,f'Ship_{sid}',cap,max_en,sp,nid))
            print(f'Ship_{sid}: px=({px},{py}) world=({wx:.0f},{wy:.0f}) → node {nid}')

# ── 读取 tasks (tasks.txt, C++像素格式) ──
tasks=[]
with open(f'{BASE}/../config/tasks.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        m=re.match(r'Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',line)
        if m:
            tid=int(m.group(1))
            ppx,ppy=int(m.group(2)),int(m.group(3))
            dpx,dpy=int(m.group(4)),int(m.group(5))
            pwx,ppy_w=ppx*2.0,ppy*2.0; dwx,dwy_w=dpx*2.0,dpy*2.0
            pu_nid=find_nearest_node(pwx,ppy_w)
            de_nid=find_nearest_node(dwx,dwy_w)
            tasks.append((tid,pu_nid,de_nid))
            print(f'Task_{tid}: px({ppx},{ppy})→({dpx},{dpy}) world({pwx:.0f},{ppy_w:.0f})→({dwx:.0f},{dwy_w:.0f}) nodes {pu_nid}→{de_nid}')

# ── 初始化调度器 ──
sh=RealTimeScheduler(rn,port_ids,gas_ids,node_names)
for sid,name,cap,en,sp,nid in ships:
    sh.add_ship(sid,name,cap,en,sp,nid)

np.random.seed(42)
for tid,pu,de in tasks:
    d=rn.dist_matrix[pu,de]
    if d>0 and d<np.inf:
        sh.add_task(tid,pu,de,float(np.random.choice([300,500,800,1000,1500])),
                   int(np.random.choice([1,2,3])),float('inf'))

sched=Scheduler(rn,sh.ships,sh.tasks,port_ids,gas_ids,node_names)

print(f'\n{"="*60}')
print(f'  Initial Scheduling: {len(sh.ships)} ships × {len(sh.tasks)} tasks')
print(f'{"="*60}')
sched.initialize()

print(f'\n{"="*60}')
print(f'  RoadNode Sequences (for path planner):')
print(f'{"="*60}')
for sid in sorted(sched.ships.keys()):
    seq=sched.get_node_sequences().get(sid,[])
    names=[node_names.get(n,f'N{n}') for n in seq]
    print(f'  {sh.ships[sid].name}: {seq}')
    print(f'    → {" → ".join(names)}')

# ── 模拟执行 ──
print(f'\n{"="*60}')
print(f'  Simulation: 60 min')
print(f'{"="*60}')
sched.run(3600,dt=300)
for sid,s in sorted(sched.ships.items()):
    node=node_names.get(s.current_node,f'N{s.current_node}')
    print(f'  {s.name} @{node} e={s.energy:.0f} load={s.load:.0f}t '
          f'done={len(s.completed_tasks)}/{len(s.task_sequence)} route={len(sched.routes.get(sid,[]))}nodes')

completed=sum(1 for t in sched.tasks.values() if t.status=='completed')
print(f'\n  Completed: {completed}/{len(sched.tasks)} tasks')
print(f'  Fleet distance: {sum(s.total_distance for s in sched.ships.values())/1000:.1f}km')
