#!/usr/bin/env python3
"""真实时间滚动优化：分批到达、固定故障、冻结已开始任务。"""
import argparse, csv, os, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
from road_network import load_road_network
from static_assignment import build_static_scene, greedy_initial_routes, solve_initial_assignment
from vns_scheduler import VNSScheduler
from tabu_scheduler import TabuScheduler
from ga_scheduler import GAScheduler
from ges_scheduler import GESScheduler
from memetic_scheduler import MemeticScheduler

ALGOS=['Greedy','Graph-ALNS','VNS','Tabu','GA','GES','Memetic']
def solve(a,rn,ships,tasks,names):
 if a=='Greedy': return greedy_initial_routes(rn,ships,tasks)
 if a=='Graph-ALNS': return solve_initial_assignment(rn,ships,tasks,names,alns_config={'use_graph_candidates':True,'use_adaptive_weights':True,'use_sa':True,'max_iter':100})
 obj={'VNS':VNSScheduler,'Tabu':TabuScheduler,'GA':GAScheduler,'GES':GESScheduler,'Memetic':MemeticScheduler}[a](rn,ships,tasks)
 # 在线滚动场景的统一计算预算，避免任一算法在每个 300s 周期耗尽时间。
 if hasattr(obj,'max_iter'): obj.max_iter=min(obj.max_iter, 20)
 if hasattr(obj,'n_generations'): obj.n_generations=min(obj.n_generations, 12)
 if hasattr(obj,'sample_size'): obj.sample_size=min(obj.sample_size, 20)
 return obj.optimize()
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=20);p.add_argument('--dt',type=float,default=300);p.add_argument('--output',default=os.getenv('TASK_PLANNER_OUTPUT'));x=p.parse_args()
 pkg=Path(__file__).resolve().parents[2];out=Path(x.output or pkg/'output');out.mkdir(exist_ok=True); ports=Path(os.getenv('TASK_PLANNER_PORTS',pkg/'config/ports.txt'));usvs=Path(os.getenv('TASK_PLANNER_USVS',pkg/'config/usvs.txt'));tf=Path(os.getenv('TASK_PLANNER_TASKS',pkg/'config/tasks.txt'))
 rn=load_road_network(str(out/'road_network.json'),str(ports));names={i:n.port_name or f'N{i}' for i,n in rn.nodes.items()};rows=[]
 print(f'Real-time rolling: {x.seeds} seeds, dt={x.dt}s, arrivals=0/1800/3600s, fault=3600s',flush=True)
 for seed in range(x.seeds):
  base=build_static_scene(rn,usvs,tf,seed=seed*1009+17); tids=sorted(base.tasks);arrival={tid:(0 if i<14 else 1800 if i<27 else 3600) for i,tid in enumerate(tids)}
  for algo in ALGOS:
   sc=deepcopy(base);pending=set();done=set();active={};failed=min(sc.ships);dist=energy=runtime=0.;replans=0;last_done=0.;faulted=False
   for now in np.arange(0,21600+x.dt,x.dt):
    for tid,t0 in arrival.items():
     if t0<=now and tid not in done and tid not in active: pending.add(tid)
    if not faulted and now>=3600:
     faulted=True
     if failed in active:
      tid,phase,remain=active.pop(failed);pending.add(tid);sc.ships[failed].load=0
     sc.ships.pop(failed,None)
    # Replan only unstarted tasks. Active tasks are excluded/frozen.
    if pending and sc.ships and int(now) % 900 == 0:
     avail={sid:s for sid,s in sc.ships.items() if sid not in active}
     if avail:
      taskmap={tid:sc.tasks[tid] for tid in pending};random.seed(seed*10000+int(now));np.random.seed(seed*10000+int(now));t=time.perf_counter();routes=solve(algo,rn,avail,taskmap,names);runtime+=time.perf_counter()-t;replans+=1
      for sid,route in routes.items():
       tid=next((n.task_id for n in route if n.action=='PICKUP'),None)
       if tid in pending:
        task=sc.tasks[tid];s=sc.ships[sid];d=rn.dist_matrix[s.current_node,task.pickup_node]
        if not np.isinf(d): active[sid]=(tid,'to_pickup',d/max(s.max_speed,1));pending.remove(tid)
    # Advance each active task through real elapsed time; unused time continues phase transitions.
    for sid in list(active):
     tid,phase,remain=active[sid];s=sc.ships[sid];task=sc.tasks[tid];left=x.dt
     while left>0 and sid in active:
      use=min(left,remain);remain-=use;left-=use
      if remain>1e-9: active[sid]=(tid,phase,remain);break
      if phase=='to_pickup':
       d=rn.dist_matrix[s.current_node,task.pickup_node];dist+=d;energy+=d/1000*s.energy_per_km;s.energy-=d/1000*s.energy_per_km;s.current_node=task.pickup_node;phase='loading';remain=300;s.load+=task.payload
      elif phase=='loading':
       d=rn.dist_matrix[s.current_node,task.delivery_node];phase='to_delivery';remain=d/max(s.max_speed,1)
      elif phase=='to_delivery':
       d=rn.dist_matrix[s.current_node,task.delivery_node];dist+=d;energy+=d/1000*s.energy_per_km;s.energy-=d/1000*s.energy_per_km;s.current_node=task.delivery_node;phase='unloading';remain=180
      else:
       s.load=max(0,s.load-task.payload);done.add(tid);last_done=now+(x.dt-left);active.pop(sid)
   rows.append({'seed':seed,'algorithm':algo,'makespan_s':last_done if len(done)==len(tids) else 21600.,'completion_rate':len(done)/len(tids),'backlog':len(tids)-len(done),'distance_m':dist,'energy_kwh':energy,'replans':replans,'runtime_s':runtime})
  print(f'  seed {seed+1}/{x.seeds} done',flush=True)
 fields=list(rows[0]);path=out/'rolling_comparison_raw.csv'
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(16,4.5));k=ALGOS
 for i,(metric,title,y) in enumerate([('makespan_s','Real-time makespan','Makespan (s)'),('completion_rate','Completion rate','Completed tasks (%)'),('runtime_s','Replanning cost','Time (s)')]):
  vals=[[r[metric]*(100 if metric=='completion_rate' else 1) for r in rows if r['algorithm']==a] for a in k];ax[i].boxplot(vals,labels=k,showmeans=True);ax[i].tick_params(axis='x',rotation=28);ax[i].set_title(title);ax[i].set_ylabel(y)
 fig.tight_layout();fig.savefig(out/'rolling_comparison.png',dpi=180,bbox_inches='tight');print(f'Wrote: {path}\nWrote: {out/"rolling_comparison.png"}')
if __name__=='__main__':main()
