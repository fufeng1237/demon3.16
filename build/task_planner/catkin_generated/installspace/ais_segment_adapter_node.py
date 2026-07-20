#!/usr/bin/env python3
"""Adapter for one AIS local-map demonstration segment.

The fleet mission remains in the whole-river ``river_map`` frame.  This node
uses explicitly configured entry/exit poses in the AIS local ``map`` frame;
it never assumes that whole-river road-network coordinates equal localmap3.
"""
import rospy
import json
import os
import yaml
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class AISSegmentAdapter:
    def __init__(self):
        rospy.init_node('ais_segment_adapter')
        self.ship_id = int(rospy.get_param('~ship_id', 0))
        self.action_index = int(rospy.get_param('~action_index', 0))
        self.use_navfn_fallback = bool(rospy.get_param('~use_navfn_fallback', False))
        self.local_ns = rospy.get_param('~local_ns', '/fleet/local/ship_{}'.format(self.ship_id)).rstrip('/')
        self.planner_param_ns = rospy.get_param('~planner_param_ns', '/ais_planner_ship_{}'.format(self.ship_id))
        self.optimizer_param_ns = rospy.get_param('~optimizer_param_ns', '/ais_optimizer_ship_{}'.format(self.ship_id))
        self.smoothed_topic = rospy.get_param('~smoothed_topic', self.local_ns + '/ais_smoothed_path')
        self.start = rospy.get_param('~local_start', [2.0, 2.0])
        self.goal = rospy.get_param('~local_goal', [70.0, 30.0])
        config = rospy.get_param('~portal_config', '')
        if config and os.path.exists(config):
            with open(config) as stream:
                self.portals = yaml.safe_load(stream).get('portals', []) or []
        else:
            self.portals = []
        self.start_pub = rospy.Publisher(self.local_ns + '/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.goal_pub = rospy.Publisher(self.local_ns + '/goal', PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher(self.local_ns + '/global_path', Path, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.local_ns + '/status', String, queue_size=1, latch=True)
        rospy.Subscriber(self.local_ns + '/request', Bool, self.request_cb)
        rospy.Subscriber('/fleet/portal_events', String, self.portal_cb)
        rospy.Subscriber('/fleet/mission_plan', String, self.mission_cb)
        # AIS is the authoritative local global planner. Navfn remains inside
        # move_base only as a DWA controller fallback and is not rendered as
        # the mission-level global path unless explicitly requested.
        if self.use_navfn_fallback:
            rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_cb)
        rospy.Subscriber(self.smoothed_topic, Path, self.path_cb)
        rospy.loginfo('AIS segment adapter ready: ship=%d action=%d local start=%s goal=%s',
                      self.ship_id, self.action_index, self.start, self.goal)
        self.bound_action = None
        self.bound_portal = None
        self.awaiting_path = False
        self.path_locked = False
        self.waypoints = []; self.waypoint_index = 0
        self.fleet_arrived_pub = rospy.Publisher('/fleet/ship_{}/arrived'.format(self.ship_id), Bool, queue_size=1)
        rospy.Subscriber(self.local_ns + '/sim_arrived', Bool, self.segment_arrived_cb)

    def mission_cb(self, msg):
        try:
            mission = json.loads(msg.data)
            actions = mission['ships'][str(self.ship_id)]['actions']
            self.bound_action = actions[self.action_index]
            route = mission['ships'][str(self.ship_id)].get('road_node_sequence', [])
            self.bound_portal = self._portal_for_route(route)
            if self.bound_portal:
                self.start = self.bound_portal['local_entry']
                self.goal = self.bound_portal['local_exit']
            self.status_pub.publish(String(data='bound ship={} action={} task={} type={} river_node={}'.format(
                self.ship_id, self.action_index, self.bound_action['task_id'],
                self.bound_action['action'], self.bound_action['node_id'])))
        except (ValueError, KeyError, IndexError) as exc:
            self.bound_action = None
            self.status_pub.publish(String(data='binding_error {}'.format(exc)))

    def _portal_for_route(self, nodes):
        for portal in self.portals:
            try:
                entry = nodes.index(portal['river_entry_node'])
                if portal['river_exit_node'] in nodes[entry + 1:]:
                    return portal
            except ValueError:
                continue
        return None

    def portal_cb(self, msg):
        try:
            event = json.loads(msg.data)
            if int(event['ship_id']) != self.ship_id:
                return
            if 'local_waypoints' in event:
                self.waypoints = event['local_waypoints']; self.waypoint_index = 0
                self.start, self.goal = self.waypoints[0], self.waypoints[1]
                direction = int(event['lane_direction'])
                self.bound_portal = {'id': '{}_to_{}'.format(event['entry_anchor'], event['exit_anchor'])}
            else:
                portal = next(p for p in self.portals if p['id'] == event['portal_id'])
                self.bound_portal = portal
                self.start, self.goal = portal['local_entry'], portal['local_exit']
                direction = int(portal.get('lane_direction', 1))
        except (ValueError, KeyError, StopIteration):
            return
        rospy.set_param(self.planner_param_ns + '/lane_direction', direction)
        rospy.set_param(self.optimizer_param_ns + '/lane_direction', direction)
        self._start_segment()

    def segment_arrived_cb(self, msg):
        if not msg.data: return
        if self.waypoints and self.waypoint_index + 2 < len(self.waypoints):
            self.waypoint_index += 1
            self.start, self.goal = self.waypoints[self.waypoint_index], self.waypoints[self.waypoint_index + 1]
            self._start_segment()
            return
        self.waypoints = []; self.fleet_arrived_pub.publish(Bool(data=True))

    def _start_segment(self):
        """Submit exactly one local segment and lock its first global path."""
        self.awaiting_path = True
        self.path_locked = False
        self.start_pub.publish(self.pose(self.start, covariance=True))
        self.goal_pub.publish(self.pose(self.goal))
        portal = self.bound_portal or {}
        self.status_pub.publish(String(data='portal_triggered id={} ship={} entry={} exit={}'.format(
            portal.get('id', 'manual_request'), self.ship_id,
            portal.get('river_entry_node', 'n/a'), portal.get('river_exit_node', 'n/a'))))

    @staticmethod
    def pose(xy, covariance=False):
        msg = PoseWithCovarianceStamped() if covariance else PoseStamped()
        msg.header.frame_id = 'map'; msg.header.stamp = rospy.Time.now()
        pose = msg.pose.pose if covariance else msg.pose
        pose.position.x, pose.position.y, pose.orientation.w = float(xy[0]), float(xy[1]), 1.0
        return msg

    def request_cb(self, msg):
        if not msg.data:
            return
        # Re-read the latched mission at request time so launch ordering cannot
        # leave the local demonstration without its whole-river binding.
        if self.bound_action is None:
            try:
                self.mission_cb(rospy.wait_for_message('/fleet/mission_plan', String, timeout=2.0))
            except rospy.ROSException:
                pass
        if self.bound_action is None:
            self.status_pub.publish(String(data='binding_warning: selected fleet action unavailable; running configured local demo segment'))
        self._start_segment()
        self.status_pub.publish(String(data='requested ship={} action={}'.format(self.ship_id, self.action_index)))

    def path_cb(self, path):
        # move_base may periodically replan.  The transition layer publishes
        # only the single path generated at portal entry, so the red reference
        # path is stable while DWA follows it.
        if not self.awaiting_path or self.path_locked or len(path.poses) < 2:
            return
        self.path_pub.publish(path)
        self.path_locked = True
        self.awaiting_path = False
        self.status_pub.publish(String(data='path_ready ship={} poses={}'.format(self.ship_id, len(path.poses))))


if __name__ == '__main__':
    AISSegmentAdapter(); rospy.spin()
