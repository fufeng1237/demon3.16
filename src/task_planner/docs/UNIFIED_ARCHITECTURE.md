# Unified task-planner architecture

The recommended workflow is `scripts.apps.run_unified_planner`.

```text
config + road_network.json
  -> domain: ShipState / TransportTask
  -> GraphALNSPlanner (heterogeneous graph + road-network candidate filtering)
  -> RoutePlan (pickup/delivery actions, versioned)
  -> route_builder (complete road-node sequence)
  -> ExecutionEngine (travel, loading, unloading, energy)
  -> RollingPlanningService (periodic/event-triggered replanning)
```

Only tasks in `pending`, `assigned`, or `at_transfer` may be re-planned.
Tasks being loaded, carried, or unloaded are frozen.  A ship fault while
carrying cargo creates a transfer state at the nearest port, changes the
task pickup node to that port, and triggers re-planning for an available ship.

The planner inserts a `REFUEL` action at a reachable gas station when the
predicted next leg would violate the configured energy reserve.  The executor
models travel to that station, refuelling service time, and restored energy.

Legacy schedulers, ROS node, demos and historical experiments are retained
for comparison but are not the recommended integration path.
