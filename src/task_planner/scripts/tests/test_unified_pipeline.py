#!/usr/bin/env python3
"""Smoke tests for the unified planning/execution/reallocation path."""
import scripts  # installs the legacy import bridge when run as a module
from road_network import load_road_network
from domain import ActionType, RouteAction, RoutePlan, ShipState, TaskState, TransportTask
from route_builder import expand_plan
from execution_engine import ExecutionEngine


def test_fault_after_pickup_becomes_transfer_task():
    rn = load_road_network('src/task_planner/output/road_network.json', 'src/task_planner/config/ports.txt')
    ports = [nid for nid, node in rn.nodes.items() if node.is_port]
    ship = ShipState(999, 'test', 2000, 10000, 8, 2.5, ports[0], 9000)
    task = TransportTask(1, ports[0], ports[1], 500)
    plan = RoutePlan(999, 1, [RouteAction(ActionType.PICKUP, ports[0], 1, 300),
                              RouteAction(ActionType.DELIVERY, ports[1], 1, 180)])
    engine = ExecutionEngine(rn, {999: ship}, {1: task}, ports)
    engine.set_plans({999: expand_plan(rn, ship, plan)})
    engine.step(300)
    engine.fail_ship(999)
    assert task.status == TaskState.AT_TRANSFER.value
    assert task.transfer_count == 1
    assert any(event.kind == 'cargo_transfer_required' for event in engine.events)


def test_refuel_action_restores_energy():
    rn = load_road_network('src/task_planner/output/road_network.json', 'src/task_planner/config/ports.txt')
    gas = next(nid for nid, node in rn.nodes.items() if node.is_gas_station)
    ship = ShipState(998, 'fuel', 2000, 10000, 8, 2.5, gas, 1000)
    engine = ExecutionEngine(rn, {998: ship}, {}, [])
    plan = RoutePlan(998, 1, [RouteAction(ActionType.REFUEL, gas, -1, 600)])
    engine.set_plans({998: expand_plan(rn, ship, plan)})
    engine.step(600)
    assert ship.energy == ship.max_energy
    assert any(event.kind == 'refueled' for event in engine.events)


if __name__ == '__main__':
    test_fault_after_pickup_becomes_transfer_task()
    test_refuel_action_restores_energy()
    print('unified pipeline smoke test passed')
