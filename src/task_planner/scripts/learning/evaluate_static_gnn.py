#!/usr/bin/env python3
"""Fair static comparison: rule Top-K ALNS vs learned Top-K ALNS."""
import argparse, json, os, random
from pathlib import Path
import numpy as np
from road_network import load_road_network
from domain import ShipState, TransportTask
from planning_service import GraphALNSPlanner
from inference import LearnedCandidateScorer
from base_scheduler import BaseScheduler, RouteNode
from static_assignment import read_configs, nearest_node


def scene(rn, usvs, task_file, seed):
    rng = np.random.default_rng(seed); ship_cfg, task_cfg = read_configs(usvs, task_file)
    ships = {sid: ShipState(sid, name, cap, energy, speed, 2.5, nearest_node(rn, px*2, py*2), energy*.9)
             for sid,name,cap,energy,speed,px,py in ship_cfg}
    tasks = {tid: TransportTask(tid, nearest_node(rn,px*2,py*2), nearest_node(rn,dx*2,dy*2),
                                float(rng.choice([300,500,800,1000,1500])), int(rng.choice([1,2,3])))
             for tid,px,py,dx,dy in task_cfg}
    nodes = [nid for nid, n in rn.nodes.items() if n.degree > 0 and not n.is_task_anchor]
    for ship in ships.values():
        if rng.random() < .75: ship.current_node = int(rng.choice(nodes))
        ship.energy *= float(rng.uniform(.65, .95))
    for task in tasks.values():
        if rng.random() < .35: task.deadline = float(rng.choice([5000,7000,9000,12000]))
    return ships, tasks


def evaluate(rn, ships, tasks, plans):
    routes = {sid: [RouteNode(a.node_id, a.action.value.upper(), a.task_id)
                    for a in p.actions if a.task_id >= 0] for sid,p in plans.items()}
    base = BaseScheduler(rn, ships, tasks)
    return {'makespan_s': base.makespan(routes), 'distance_m': base.total_distance(routes),
            'energy_kwh': base.total_energy(routes), 'load_std': base.load_std(routes)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--seeds',type=int,default=10);p.add_argument('--max-iter',type=int,default=100);p.add_argument('--k',type=int,default=3);p.add_argument('--output',default='gnn_static_evaluation.json');p.add_argument('--tasks',default=os.getenv('TASK_PLANNER_TASKS'));a=p.parse_args()
    package=Path(__file__).resolve().parents[2];rn=load_road_network(str(package/'output/road_network.json'),str(package/'config/ports.txt'))
    names={i:n.port_name or f'N{i}' for i,n in rn.nodes.items()}; gas=[i for i,n in rn.nodes.items() if n.is_gas_station]
    tasks_path=Path(a.tasks) if a.tasks else package/'config/tasks.txt'; results={'rule':[],'hgt':[]}
    for seed in range(a.seeds):
        for key,model in [('rule',None),('hgt',a.model)]:
            random.seed(seed);np.random.seed(seed);ships,tasks=scene(rn,package/'config/usvs.txt',tasks_path,seed)
            planner=GraphALNSPlanner(rn,names,max_iter=a.max_iter,k_candidates=a.k,gas_ids=gas)
            if model: planner.candidate_scorer=LearnedCandidateScorer(model,k=a.k)
            results[key].append(evaluate(rn,ships,tasks,planner.plan(ships,tasks)))
        print(f'{seed+1}/{a.seeds}',flush=True)
    summary={k:{m:float(np.mean([x[m] for x in vals])) for m in vals[0]} for k,vals in results.items()}
    Path(a.output).write_text(json.dumps({'tasks_path':str(tasks_path),'raw':results,'mean':summary},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
