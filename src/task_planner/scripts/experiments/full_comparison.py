#!/usr/bin/env python3
"""完整对比: 8船×40任务 — Greedy vs NN vs Plain ALNS vs Graph-ALNS"""
import sys,os,re,time,itertools,numpy as np,heapq
from copy import deepcopy
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from scheduler import Scheduler
from real_time_scheduler import RealTimeScheduler
from alns_scheduler import RouteNode, ALNSScheduler
from graph_evaluator import GraphEvaluator

BASE=os.path.dirname(os.path.abspath(__file__))
rn=load_road_network(f'{BASE}/../../output/road_network.json',f'{BASE}/../../config/ports.yaml')
port_ids=sorted([nid for nid,n in rn.nodes.items() if n.is_port])
gas_ids=sorted([nid for nid,n in rn.nodes.items() if n.is_gas_station])
node_names={nid:(n.port_name if n.port_name else f'N{nid}') for nid,n in rn.nodes.items()}

def nn(wx,wy):
    best,best_d=None,float('inf')
    for nid,n in rn.nodes.items():
        d=np.sqrt((n.x-wx)**2+(n.y-wy)**2)
        if d<best_d: best_d=d; best=nid
    return best

# Read configs
ship_cfg=[]; task_cfg=[]
with open(f'{BASE}/../../config/usvs.txt') as f:
    for l in f:
        if l[0]=='#': continue
        parts=l.replace('USV:','').strip().split(',')
        if len(parts)>=8:
            sid=int(parts[0]); px=int(parts[1]); py=int(parts[2])
            ship_cfg.append((sid,f'S{sid}',int(parts[4]),float(parts[6]),float(parts[3]),px,py))
with open(f'{BASE}/../../config/tasks.txt') as f:
    for l in f:
        if l[0]=='#': continue
        m=re.match(r'Task\s+(\d+):\s*pickup\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\).*delivery\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',l)
        if m: task_cfg.append((int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5))))

n_ships=len(ship_cfg); n_tasks=len(task_cfg)  # 自动读取全部
u_ships=ship_cfg[:n_ships]; u_tasks=task_cfg[:n_tasks]

def build_scene():
    sh=RealTimeScheduler(rn,port_ids,gas_ids,node_names)
    for sid,name,cap,en,sp,px,py in u_ships:
        sh.add_ship(sid,name,cap,en,sp,nn(px*2,py*2))
    np.random.seed(42)
    for tid,ppx,ppy,dpx,dpy in u_tasks:
        sh.add_task(tid,nn(ppx*2,ppy*2),nn(dpx*2,dpy*2),float(np.random.choice([300,500,800,1000,1500])),int(np.random.choice([1,2,3])),float('inf'))
    return sh

def simulate(sh,assign):
    totd=0; maxt=0; loads=[]; ship_t={}
    for sid in assign:
        ship_t[sid]=0.0; cur=sh.ships[sid].current_node; ld=0
        for tid in assign[sid]:
            tk=sh.tasks[tid]
            d1=rn.dist_matrix[cur,tk.pickup_node]; d2=rn.dist_matrix[tk.pickup_node,tk.delivery_node]
            ship_t[sid]+=d1/max(sh.ships[sid].max_speed,1)+300+d2/max(sh.ships[sid].max_speed,1)+180
            totd+=d1+d2; cur=tk.delivery_node; ld+=tk.payload
        maxt=max(maxt,ship_t[sid]); loads.append(ld)
    return maxt,totd,np.std(loads),loads

# ====== 4 METHODS ======
results={}
for method in ['Greedy','NearestNeighbor','Plain ALNS','Graph-ALNS']:
    results[method]={'makespan':[],'dist':[],'load_std':[],'time':[],'tasks_per_ship':[]}
    for seed in range(5):
        sh=build_scene(); np.random.seed(seed*10+42)
        for tid in sh.tasks: sh.tasks[tid].payload=float(np.random.choice([300,500,800,1000,1500]))
        t0=time.perf_counter()
        if method=='Greedy':
            for tid in u_tasks:
                t=sh.tasks[tid[0]]
                best_sid,best_d=None,float('inf')
                for sid,s in sh.ships.items():
                    d=rn.dist_matrix[s.current_node,t.pickup_node]+rn.dist_matrix[t.pickup_node,t.delivery_node]
                    if d<best_d and t.payload<=s.remaining_capacity: best_d=d; best_sid=sid
                if best_sid: sh.ships[best_sid].task_sequence.append(tid[0])
            assign={sid:list(s.task_sequence) for sid,s in sh.ships.items()}
        elif method=='NearestNeighbor':
            unassigned=set(range(n_tasks)); ship_order=sorted(sh.ships.keys())
            for sid in ship_order:
                cur=sh.ships[sid].current_node
                while unassigned:
                    best_tid,best_d=None,float('inf')
                    for tid in unassigned:
                        t=sh.tasks[tid]; d=rn.dist_matrix[cur,t.pickup_node]+rn.dist_matrix[t.pickup_node,t.delivery_node]
                        if d<best_d and t.payload<=sh.ships[sid].remaining_capacity: best_d=d; best_tid=tid
                    if best_tid is None: break
                    sh.ships[sid].task_sequence.append(best_tid); cur=sh.tasks[best_tid].delivery_node; unassigned.remove(best_tid)
            assign={sid:list(s.task_sequence) for sid,s in sh.ships.items()}
        elif method=='Plain ALNS':
            evaluator=GraphEvaluator(rn,node_names)
            alns=ALNSScheduler(evaluator,sh.tasks,rn,node_names); alns.K_candidates=len(sh.ships)
            routes=alns.build_initial_routes(sh.ships); routes=alns.optimize(sh.ships,routes)
            assign={}
            for sid in sh.ships:
                tids=[]; route=routes.get(sid,[])
                for rnd in route:
                    if rnd.action=="PICKUP" and rnd.task_id>=0: tids.append(rnd.task_id)
                assign[sid]=tids
        elif method=='Graph-ALNS':
            sched=Scheduler(rn,sh.ships,sh.tasks,port_ids,gas_ids,node_names)
            sched.initialize()
            assign={}
            for sid in sh.ships:
                tids=[]; route=sched.routes.get(sid,[])
                for rnd in route:
                    if rnd.action=="PICKUP" and rnd.task_id>=0: tids.append(rnd.task_id)
                assign[sid]=tids
        results[method]['time'].append(time.perf_counter()-t0)
        ms,td,lsd,loads=simulate(sh,assign)
        results[method]['makespan'].append(ms); results[method]['dist'].append(td)
        results[method]['load_std'].append(lsd)
        results[method]['tasks_per_ship'].append([len(assign.get(s,[])) for s in sorted(sh.ships.keys())])

    print(f'{method}: makespan={np.mean(results[method]["makespan"]):.0f}s dist={np.mean(results[method]["dist"]):.0f}m time={np.mean(results[method]["time"]):.3f}s')

# ====== PRINT TABLE ======
print(f'\n{"Method":<20} {"Makespan":>10} {"Distance":>10} {"LoadStd":>8} {"Time(s)":>8} {"Tasks/Ship"}')
print("-"*80)
m_names=['Greedy','NearestNeighbor','Plain ALNS','Graph-ALNS']
for m in m_names:
    r=results[m]
    ms=np.mean(r['makespan']); td=np.mean(r['dist']); ls=np.mean(r['load_std']); tm=np.mean(r['time'])
    ts=[np.mean([r['tasks_per_ship'][i][j] for i in range(5)]) for j in range(n_ships)]
    ts_str=' '.join(f'{t:.0f}' for t in ts)
    print(f'{m:<20} {ms:10.0f} {td:10.0f} {ls:8.0f} {tm:8.3f}  {ts_str}')

# ====== CHART ======
fig=plt.figure(figsize=(22,14))
gs=fig.add_gridspec(3,3,hspace=0.35,wspace=0.3)

for idx,(m,lbl) in enumerate([('makespan','Makespan (s)'),('dist','Total Distance (m)'),('load_std','Load StdDev')]):
    ax=fig.add_subplot(gs[0,idx])
    vals=[np.mean(results[n][m]) for n in m_names]
    colors=['#ff6666','#ffaa44','#44aaff','#33cc33']
    ax.bar(m_names,vals,color=colors)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*0.02,f'{v:.0f}',ha='center',fontsize=9)
    ax.set_title(lbl); ax.tick_params(rotation=15)

# Time
ax=fig.add_subplot(gs[1,0])
vals=[np.mean(results[n]['time']) for n in m_names]
ax.bar(m_names,vals,color=colors)
for i,v in enumerate(vals): ax.text(i,v+0.01,f'{v:.3f}s',ha='center',fontsize=8)
ax.set_title('Algorithm Time'); ax.tick_params(rotation=15)

# Task distribution
ax=fig.add_subplot(gs[1,1:])
x=np.arange(n_ships); w=0.2
for i,(m,c) in enumerate(zip(m_names,colors)):
    ts=[np.mean([results[m]['tasks_per_ship'][j][k] for j in range(5)]) for k in range(n_ships)]
    ax.bar(x+i*w-w,ts,w,label=m,color=c,alpha=0.8)
ax.set_xticks(x); ax.set_xlabel('Ship'); ax.set_ylabel('Tasks'); ax.set_title('Per-Ship Task Distribution'); ax.legend()

# Per-ship distance breakdown for Graph-ALNS
ax=fig.add_subplot(gs[2,:])
# Run one more time to get detailed per-ship metrics
sh_detail=build_scene()
sched_detail=Scheduler(rn,sh_detail.ships,sh_detail.tasks,port_ids,gas_ids,node_names)
sched_detail.initialize()
ship_details=[]
for sid in sorted(sh_detail.ships.keys()):
    s=sh_detail.ships[sid]; route=sched_detail.routes.get(sid,[])
    n_t=len([r for r in route if r.action=='PICKUP'])
    tot_d=0; cur=s.current_node
    for rnd in route:
        tot_d+=rn.dist_matrix[cur,rnd.node_id]; cur=rnd.node_id
    ship_details.append((s.name,n_t,tot_d))
names=[d[0] for d in ship_details]; nts=[d[1] for d in ship_details]; tds=[d[2] for d in ship_details]
ax2=ax.twinx()
b1=ax.bar(names,nts,color='#3388ff',alpha=0.7,label='Tasks')
b2=ax2.bar(names,tds,color='#ff6644',alpha=0.7,label='Distance(m)',width=0.4,align='edge')
ax.set_ylabel('Task Count'); ax2.set_ylabel('Distance (m)')
ax.set_title('Graph-ALNS: Per-Ship Task Count & Distance')
lines1,labels1=ax.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,labels1+labels2)
for i,(nt,td) in enumerate(zip(nts,tds)):
    ax.text(i,nt+0.2,f'{nt}',ha='center',fontsize=9)
    ax2.text(i+0.2,td+20,f'{td:.0f}m',ha='center',fontsize=8,color='#ff6644')

fig.suptitle(f'Algorithm Comparison: {n_ships} Ships x {n_tasks} Tasks (5 seeds avg)',fontsize=14,fontweight='bold')
plt.savefig(f'{BASE}/../../output/full_comparison.png',dpi=120,bbox_inches='tight'); plt.close()
print(f'\nSaved: output/full_comparison.png')

# ====== ALLOCATION MAP with detailed routes ======
img=np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
if img.ndim==3: img=img[:,:,0]
h_px,w_px=img.shape; ps=2.0; wh=h_px*ps; ww=w_px*ps
bg=np.flipud(img); water=bg>127; wr,wc=np.where(water); m=50

fig,ax=plt.subplots(figsize=(26,13))
ax.imshow(bg,extent=[0,ww,0,wh],origin='lower',cmap='gray')
for e in rn.edges:
    n1,n2=rn.nodes[e.from_id],rn.nodes[e.to_id]
    ax.plot([n1.x,n2.x],[wh-n1.y,wh-n2.y],'cyan',lw=0.3,alpha=0.2,zorder=1)
for n in rn.nodes.values():
    if n.is_port: ax.scatter(n.x,wh-n.y,c='red',s=30,marker='s',edgecolors='white',lw=0.5,zorder=3)

colors=['#ff3333','#3388ff','#33cc33','#ff9933','#ff44ff','#44ffff','#ffff44','#ff8844']
path_cache={}
def get_path(fr,to):
    k=(fr,to)
    if k in path_cache: return path_cache[k]
    dist={fr:0}; prev={}; q=[(0,fr)]
    while q:
        d,u=heapq.heappop(q)
        if d>dist.get(u,float('inf')): continue
        if u==to: break
        for v in rn.adj.get(u,[]):
            w=rn.dist_matrix[u,v]; nd=d+w
            if nd<dist.get(v,float('inf')): dist[v]=nd; prev[v]=u; heapq.heappush(q,(nd,v))
    if to not in prev: path=[fr,to]
    else:
        path=[]; cur=to
        while cur!=fr: path.append(cur); cur=prev[cur]
        path.append(fr); path.reverse()
    path_cache[k]=path; return path

legend_items=[]
# Text box for route details
detail_lines=["Route Details (Graph-ALNS):",""]
for i,sid in enumerate(sorted(sh_detail.ships.keys())):
    route=sched_detail.routes.get(sid,[])
    ship=sh_detail.ships[sid]; cur=ship.current_node; pts=[]
    node=rn.nodes.get(cur)
    if node: pts.append([node.x,wh-node.y])
    last_node=cur
    for rnode in route:
        path=get_path(cur,rnode.node_id)
        for nid in path[1:]:
            nd=rn.nodes.get(nid)
            if nd: pts.append([nd.x,wh-nd.y])
        cur=rnode.node_id
    if len(pts)>1:
        xs,ys=zip(*pts)
        ax.plot(xs,ys,color=colors[i],lw=3,alpha=0.8,zorder=5)
        n_t=len([r for r in route if r.action=='PICKUP'])
        tot_d=sum(rn.dist_matrix[u][v] for u,v in zip([last_node]+[r.node_id for r in route[:-1]],[r.node_id for r in route]))
        last_node=cur
        # Route detail text
        pu_nodes=[(r.node_id,rn.nodes[r.node_id].port_name if rn.nodes[r.node_id].port_name else f'N{r.node_id}') for r in route if r.action=='PICKUP']
        de_nodes=[(r.node_id,rn.nodes[r.node_id].port_name if rn.nodes[r.node_id].port_name else f'N{r.node_id}') for r in route if r.action=='DELIVERY']
        detail_lines.append(f'{ship.name}: {n_t} tasks, {len(route)} nodes')
        for j in range(min(n_t,4)):
            if j<len(pu_nodes) and j<len(de_nodes):
                detail_lines.append(f'  {pu_nodes[j][1]}→{de_nodes[j][1]}')
        if n_t>4: detail_lines.append(f'  ...+{n_t-4} more')
        legend_items.append(Patch(color=colors[i],label=f'{ship.name}: {n_t} tasks'))

for i,sid in enumerate(sorted(sh_detail.ships.keys())):
    node=rn.nodes.get(sh_detail.ships[sid].current_node)
    if node:
        ax.scatter(node.x,wh-node.y,c=colors[i],s=300,marker='*',edgecolors='white',lw=2,zorder=7)

# Detail text box
detail_text='\n'.join(detail_lines)
ax.text(0.02,0.98,detail_text,transform=ax.transAxes,fontsize=8,fontfamily='monospace',
        va='top',color='white',bbox=dict(boxstyle='round',facecolor='#111',alpha=0.85))

ax.set_xlim(max(0,wc.min()-m)*ps,min(w_px,wc.max()+m)*ps)
ax.set_ylim(max(0,wr.min()-m)*ps,min(h_px,wr.max()+m)*ps)
ax.set_aspect('equal')
ax.legend(handles=legend_items,fontsize=8,loc='upper right')
ax.set_title(f'Graph-ALNS Route Allocation: {n_ships} Ships x {n_tasks} Tasks (Road Network Paths)',
             fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{BASE}/../../output/allocation_routes.png',dpi=150,bbox_inches='tight',facecolor='black'); plt.close()
print('Saved: output/allocation_routes.png')
