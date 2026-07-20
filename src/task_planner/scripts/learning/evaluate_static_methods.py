#!/usr/bin/env python3
"""Unified fair static comparison including learned Graph+ALNS and classics."""
import argparse, copy, json, os, random
from pathlib import Path
import numpy as np
from evaluate_static_gnn import scene, evaluate
from road_network import load_road_network
from planning_service import GraphALNSPlanner
from inference import LearnedCandidateScorer
from base_scheduler import BaseScheduler
from vns_scheduler import VNSScheduler
from tabu_scheduler import TabuScheduler
from ga_scheduler import GAScheduler
from ges_scheduler import GESScheduler
from memetic_scheduler import MemeticScheduler


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--seeds',type=int,default=3);p.add_argument('--iter',type=int,default=100);p.add_argument('--k',type=int,default=3);p.add_argument('--output',default='static_methods.json');p.add_argument('--tasks',default=os.getenv('TASK_PLANNER_TASKS'));a=p.parse_args()
 pkg=Path(__file__).resolve().parents[2];rn=load_road_network(str(pkg/'output/road_network.json'),str(pkg/'config/ports.txt')); names={i:n.port_name or f'N{i}' for i,n in rn.nodes.items()};gas=[i for i,n in rn.nodes.items() if n.is_gas_station]
 tasks_path=Path(a.tasks) if a.tasks else pkg/'config/tasks.txt'; methods=['Greedy','VNS','Tabu','GA','GES','Memetic','HGT-ALNS'];result={m:[] for m in methods}
 for seed in range(a.seeds):
  base=scene(rn,pkg/'config/usvs.txt',tasks_path,seed)
  for name in methods:
   random.seed(seed);np.random.seed(seed);ships,tasks=copy.deepcopy(base)
   if name=='Greedy': routes=BaseScheduler(rn,ships,tasks).greedy_init()
   elif name=='HGT-ALNS':
    planner=GraphALNSPlanner(rn,names,max_iter=a.iter,k_candidates=a.k,gas_ids=gas);planner.candidate_scorer=LearnedCandidateScorer(a.model,k=a.k);plans=planner.plan(ships,tasks);result[name].append(evaluate(rn,ships,tasks,plans));continue
   else:
    cls={'VNS':VNSScheduler,'Tabu':TabuScheduler,'GA':GAScheduler,'GES':GESScheduler,'Memetic':MemeticScheduler}[name];alg=cls(rn,ships,tasks)
    if hasattr(alg,'max_iter'):alg.max_iter=min(alg.max_iter,a.iter)
    if hasattr(alg,'n_generations'):alg.n_generations=min(alg.n_generations,max(10,a.iter//2))
    routes=alg.optimize()
   b=BaseScheduler(rn,ships,tasks);result[name].append({'makespan_s':b.makespan(routes),'distance_m':b.total_distance(routes),'energy_kwh':b.total_energy(routes),'load_std':b.load_std(routes)})
  print(f'{seed+1}/{a.seeds}',flush=True)
 mean={m:{k:float(np.mean([x[k] for x in vals])) for k in vals[0]} for m,vals in result.items()};Path(a.output).write_text(json.dumps({'tasks_path':str(tasks_path),'raw':result,'mean':mean},indent=2));print(json.dumps(mean,indent=2))
if __name__=='__main__':main()
