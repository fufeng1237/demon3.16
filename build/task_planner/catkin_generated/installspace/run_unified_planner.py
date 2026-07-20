#!/usr/bin/env python3
"""Recommended task_planner entrypoint: static plan -> execution -> rolling replan."""
import argparse, json, os
from pathlib import Path
from domain import ShipState, TransportTask
from road_network import load_road_network
from static_assignment import read_configs, nearest_node
from planning_service import GraphALNSPlanner
from execution_engine import ExecutionEngine
from rolling_service import RollingPlanningService
from inference import LearnedCandidateScorer


def build_domain_scene(rn, usvs_path, tasks_path):
    ship_cfg, task_cfg = read_configs(usvs_path, tasks_path)
    ships, tasks = {}, {}
    for sid, name, cap, energy, speed, px, py in ship_cfg:
        ships[sid] = ShipState(sid, name, cap, energy, speed, 2.5,
                               nearest_node(rn, px * 2, py * 2), energy * .9)
    for tid, px, py, dx, dy in task_cfg:
        tasks[tid] = TransportTask(tid, nearest_node(rn, px*2, py*2),
                                   nearest_node(rn, dx*2, dy*2), 800.0, priority=1)
    return ships, tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=7200)
    p.add_argument('--dt', type=float, default=300)
    p.add_argument('--fault-time', type=float, default=-1)
    p.add_argument('--fault-ship', type=int, default=0)
    p.add_argument('--output', default=os.getenv('TASK_PLANNER_OUTPUT'))
    p.add_argument('--tasks', default=os.getenv('TASK_PLANNER_TASKS'),
                   help='task config path; defaults to TASK_PLANNER_TASKS or config/tasks.txt')
    p.add_argument('--gnn-model', default='', help='optional trained HGT checkpoint')
    a = p.parse_args()
    package = Path(__file__).resolve().parents[2]
    output = Path(a.output or package / 'output'); output.mkdir(exist_ok=True)
    rn = load_road_network(str(output/'road_network.json'), str(package/'config/ports.txt'))
    tasks_path = Path(a.tasks) if a.tasks else package/'config/tasks.txt'
    ships, tasks = build_domain_scene(rn, package/'config/usvs.txt', tasks_path)
    names = {nid: n.port_name or f'N{nid}' for nid, n in rn.nodes.items()}
    ports = [nid for nid, n in rn.nodes.items() if n.is_port]
    gas = [nid for nid, n in rn.nodes.items() if n.is_gas_station]
    engine = ExecutionEngine(rn, ships, tasks, ports)
    planner = GraphALNSPlanner(rn, names, gas_ids=gas)
    if a.gnn_model:
        planner.candidate_scorer = LearnedCandidateScorer(a.gnn_model)
    service = RollingPlanningService(planner, engine)
    service.replan('initial')
    while engine.time < a.duration:
        if a.fault_time >= 0 and engine.time >= a.fault_time and not ships.get(a.fault_ship).failed:
            service.inject_fault(a.fault_ship)
        service.step(a.dt)
    payload = {
        'time': engine.time, 'tasks_path': str(tasks_path), 'plan_history': service.plan_history,
        'events': [e.__dict__ for e in engine.events],
        'tasks': {tid: {'status': t.status, 'assigned_ship': t.assigned_ship,
                        'transfer_count': t.transfer_count} for tid, t in tasks.items()},
        'plans': {sid: {'version': p.version, 'road_node_sequence': p.node_sequence(),
                        'actions': [{'type': x.action.value, 'node_id': x.node_id, 'task_id': x.task_id}
                                    for x in p.actions]} for sid, p in engine.plans.items()}
    }
    path = output/'unified_planner_result.json'; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'Wrote: {path}')


if __name__ == '__main__':
    main()
