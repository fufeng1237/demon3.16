#!/usr/bin/env python3
"""ROS bridge: unified task plans -> per-ship AIS global-planning requests.

This node deliberately does not modify ais_navigation.  A local planner (or a
simple simulator) acknowledges each completed leg through ``~ship_<id>/arrived``.
"""
import os
import json
import sys
from pathlib import Path as FilePath
import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

# Catkin installs legacy flat-module wrappers with names such as
# ``road_network.py``.  Prefer the grouped source modules for this new node.
_SCRIPTS = FilePath(__file__).resolve().parents[1]
for _part in ('core', 'algorithms', 'learning', 'runtime'):
    _path = str(_SCRIPTS / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from road_network import load_road_network
from static_assignment import read_configs, nearest_node
from domain import ShipState, TransportTask, ActionType
from planning_service import GraphALNSPlanner
from inference import LearnedCandidateScorer
from route_builder import expand_plan


class FleetDemoBridge:
    def __init__(self):
        rospy.init_node('fleet_demo_bridge')
        pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        output = rospy.get_param('~output_dir', os.path.join(pkg, 'output'))
        tasks_path = rospy.get_param('~tasks_path', os.path.join(pkg, 'config', 'tasks.txt'))
        usvs_path = rospy.get_param('~usvs_path', os.path.join(pkg, 'config', 'usvs.txt'))
        model = rospy.get_param('~gnn_model', '')
        max_iter = int(rospy.get_param('~max_iter', 100))
        self.frame = rospy.get_param('~frame_id', 'river_map')
        # Road-network y coordinates originate at the top of the source image;
        # ROS map coordinates originate at the bottom.
        self.river_height = float(rospy.get_param('~river_height', 3252.0))
        # Whole-river allocation is independent of the AIS local-map demo.
        # Keep false unless a same-map integration is deliberately configured.
        self.dispatch_to_ais = bool(rospy.get_param('~dispatch_to_ais', False))
        self.rn = load_road_network(os.path.join(output, 'road_network.json'),
                                    os.path.join(pkg, 'config', 'ports.txt'))
        self.ships, self.tasks = self._scene(usvs_path, tasks_path)
        names = {nid: n.port_name or 'N{}'.format(nid) for nid, n in self.rn.nodes.items()}
        gas = [nid for nid, n in self.rn.nodes.items() if n.is_gas_station]
        planner = GraphALNSPlanner(self.rn, names, max_iter=max_iter, k_candidates=4, gas_ids=gas)
        if model:
            planner.candidate_scorer = LearnedCandidateScorer(model, k=4)
        self.plans = planner.plan(self.ships, self.tasks)
        # Allocation actions are high-level ports.  Expand them into actual
        # road-network node sequences before any RViz/execution publication.
        for sid, plan in self.plans.items():
            self.plans[sid] = expand_plan(self.rn, self.ships[sid], plan)
        self.action_idx = {sid: 0 for sid in self.ships}
        self.goal_pub, self.start_pub, self.path_pub = {}, {}, {}
        self.markers = rospy.Publisher('/fleet/rviz_markers', MarkerArray, queue_size=1, latch=True)
        self.plan_pub = rospy.Publisher('/fleet/mission_plan', String, queue_size=1, latch=True)
        for sid in self.ships:
            ns = '/fleet/ship_{}'.format(sid)
            self.start_pub[sid] = rospy.Publisher(ns + '/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
            self.goal_pub[sid] = rospy.Publisher(ns + '/goal', PoseStamped, queue_size=1, latch=True)
            self.path_pub[sid] = rospy.Publisher(ns + '/global_path', Path, queue_size=1, latch=True)
            rospy.Subscriber(ns + '/ais_path', Path, self._path_cb, callback_args=sid)
            rospy.Subscriber(ns + '/arrived', Bool, self._arrived_cb, callback_args=sid)
        self._publish_plan(); self._publish_global_paths(); self._publish_markers()
        if self.dispatch_to_ais:
            for sid in self.ships: self._request_leg(sid)
        rospy.loginfo('Fleet bridge ready: %d ships, %d tasks, tasks=%s', len(self.ships), len(self.tasks), tasks_path)

    def _scene(self, usvs_path, tasks_path):
        ship_cfg, task_cfg = read_configs(usvs_path, tasks_path); ships = {}; tasks = {}
        for sid, name, cap, energy, speed, px, py in ship_cfg:
            ships[sid] = ShipState(sid, name, cap, energy, speed, 2.5,
                                   nearest_node(self.rn, px * 2, py * 2), energy * .9)
        for tid, px, py, dx, dy in task_cfg:
            tasks[tid] = TransportTask(tid, nearest_node(self.rn, px*2, py*2),
                                       nearest_node(self.rn, dx*2, dy*2), 800.0, priority=1)
        return ships, tasks

    def _pose(self, nid, stamped=False):
        node = self.rn.nodes[nid]
        if stamped:
            msg = PoseWithCovarianceStamped(); pose = msg.pose.pose
        else:
            msg = PoseStamped(); pose = msg.pose
        msg.header.frame_id = self.frame; msg.header.stamp = rospy.Time.now()
        pose.position.x, pose.position.y, pose.orientation.w = node.x, node.y, 1.0
        return msg

    def _request_leg(self, sid):
        plan = self.plans.get(sid); index = self.action_idx[sid]
        if not plan or index >= len(plan.actions): return
        self.start_pub[sid].publish(self._pose(self.ships[sid].current_node, stamped=True))
        self.goal_pub[sid].publish(self._pose(plan.actions[index].node_id))

    def _path_cb(self, path, sid):
        path.header.frame_id = self.frame
        self.path_pub[sid].publish(path)

    def _arrived_cb(self, message, sid):
        if not message.data: return
        plan = self.plans.get(sid); index = self.action_idx[sid]
        if not plan or index >= len(plan.actions): return
        action = plan.actions[index]; self.ships[sid].current_node = action.node_id
        task = self.tasks.get(action.task_id)
        if task:
            if action.action == ActionType.PICKUP: task.status = 'to_delivery'
            elif action.action == ActionType.DELIVERY: task.status = 'completed'
        self.action_idx[sid] += 1; self._publish_plan(); self._publish_markers()
        if self.dispatch_to_ais:
            self._request_leg(sid)

    def _publish_plan(self):
        payload = {'frame_id': self.frame, 'ships': {
            str(sid): {'current_node': self.ships[sid].current_node,
                       'next_action_index': self.action_idx[sid],
                       'road_node_sequence': self._road_nodes(sid, plan),
                       'actions': [{'action': a.action.value, 'task_id': a.task_id,
                                    'node_id': a.node_id} for a in plan.actions]}
            for sid, plan in self.plans.items()},
            'tasks': {str(tid): {'status': task.status, 'assigned_ship': task.assigned_ship}
                      for tid, task in self.tasks.items()}}
        self.plan_pub.publish(String(data=json.dumps(payload)))

    def _publish_global_paths(self):
        """Expose each complete assigned road-node sequence to RViz/executors."""
        now = rospy.Time.now()
        for sid, plan in self.plans.items():
            sequence = self._road_nodes(sid, plan)
            path = Path(); path.header.frame_id = self.frame; path.header.stamp = now
            previous = None
            for nid in sequence:
                if nid == previous or nid not in self.rn.nodes:
                    continue
                previous = nid; node = self.rn.nodes[nid]
                pose = PoseStamped(); pose.header.frame_id = self.frame; pose.header.stamp = now
                pose.pose.position.x = node.x
                pose.pose.position.y = self.river_height - node.y
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.path_pub[sid].publish(path)

    def _road_nodes(self, sid, plan):
        """Return the continuous road path, not straight task-to-task chords."""
        nodes = [self.ships[sid].current_node]
        for action in plan.actions:
            leg = action.road_nodes or [action.node_id]
            nodes.extend(leg[1:] if leg and leg[0] == nodes[-1] else leg)
        return nodes

    def _publish_markers(self):
        arr = MarkerArray(); now = rospy.Time.now(); mid = 0
        # RViz requires a valid quaternion even for list/line markers whose pose
        # is otherwise unused.  Supplying identity prevents it from discarding
        # the visualisation with an "Uninitialized quaternion" warning.
        def marker():
            result = Marker()
            result.pose.orientation.w = 1.0
            return result

        def point(node, z=0.0):
            return Point(node.x, self.river_height - node.y, z)

        # Whole-river road network.
        road = marker(); road.header.frame_id = self.frame; road.header.stamp = now
        road.ns = 'road_network'; road.id = mid; mid += 1; road.type = Marker.LINE_LIST; road.action = Marker.ADD
        # Keep the base road network visible beneath the per-vessel routes.
        road.scale.x = 5.0; road.color.r = .15; road.color.g = .75; road.color.b = .95; road.color.a = .95
        for edge in self.rn.edges:
            a, b = self.rn.nodes[edge.from_id], self.rn.nodes[edge.to_id]
            road.points.extend([point(a, 0.15), point(b, 0.15)])
        arr.markers.append(road)
        pickups = marker(); pickups.header.frame_id = self.frame; pickups.header.stamp = now
        pickups.ns = 'pickups'; pickups.id = mid; mid += 1; pickups.type = Marker.SPHERE_LIST; pickups.action = Marker.ADD
        pickups.scale.x = pickups.scale.y = pickups.scale.z = 40; pickups.color.g = 1; pickups.color.r = .9; pickups.color.a = 1
        deliveries = marker(); deliveries.header.frame_id = self.frame; deliveries.header.stamp = now
        deliveries.ns = 'deliveries'; deliveries.id = mid; mid += 1; deliveries.type = Marker.CUBE_LIST; deliveries.action = Marker.ADD
        deliveries.scale.x = deliveries.scale.y = deliveries.scale.z = 40; deliveries.color.r = 1; deliveries.color.a = 1
        for task in self.tasks.values():
            p, d = self.rn.nodes[task.pickup_node], self.rn.nodes[task.delivery_node]
            pickups.points.append(point(p, 1)); deliveries.points.append(point(d, 1))
        arr.markers.extend([pickups, deliveries])
        # Full assigned road-node sequence, one colour per ship.
        for sid, plan in self.plans.items():
            route = marker(); route.header.frame_id = self.frame; route.header.stamp = now
            route.ns = 'assigned_routes'; route.id = mid; mid += 1; route.type = Marker.LINE_STRIP; route.action = Marker.ADD
            route.scale.x = 9.0; route.color.r = (sid * .31) % 1; route.color.g = .8; route.color.b = 1 - route.color.r; route.color.a = .9
            route.points = [point(self.rn.nodes[nid], 2) for nid in self._road_nodes(sid, plan)]
            if len(route.points) > 1: arr.markers.append(route)
        for sid, ship in self.ships.items():
            n = self.rn.nodes[ship.current_node]; m = marker(); m.header.frame_id = self.frame; m.header.stamp = now
            m.ns = 'ships'; m.id = mid; mid += 1; m.type = Marker.SPHERE; m.action = Marker.ADD; m.pose.position = point(n); m.pose.orientation.w = 1
            m.scale.x = m.scale.y = 60; m.scale.z = 18; m.color.r = (sid * .31) % 1; m.color.g = .8; m.color.b = 1 - m.color.r; m.color.a = 1; arr.markers.append(m)
            # Static allocation is conveyed by the coloured assigned routes.
            # Do not add execution-step/energy labels here: they duplicate the
            # moving-ship layer and make the whole-river view unreadable.
        self.markers.publish(arr)


if __name__ == '__main__':
    FleetDemoBridge(); rospy.spin()
