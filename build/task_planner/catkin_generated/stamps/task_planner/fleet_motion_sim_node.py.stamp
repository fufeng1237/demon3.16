#!/usr/bin/env python3
"""Visual whole-river fleet execution simulator driven by allocated paths."""
import math
import json
import os
import rospy
import yaml
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, String


class FleetMotionSim:
    def __init__(self):
        rospy.init_node('fleet_motion_sim')
        self.frame = rospy.get_param('~frame_id', 'river_map')
        self.speed = float(rospy.get_param('~speed', 45.0))
        self.ships = int(rospy.get_param('~ships', 8))
        config = rospy.get_param('~portal_config', '')
        if config and os.path.exists(config):
            with open(config) as stream:
                cfg = yaml.safe_load(stream) or {}; self.portals = cfg.get('portals', []) or []; self.anchors = cfg.get('anchors', []) or []
        else:
            self.portals = []; self.anchors = []
        self.triggered_portals = set()
        self.local_mode = set()
        self.completed = set(); self.all_complete_announced = False
        self.portal_exit_progress = {}
        self.pending_local = {}; self.local_done = set()
        self.paths = {}; self.progress = {}; self.traces = {}
        self.pub = rospy.Publisher('/fleet/motion_markers', MarkerArray, queue_size=1)
        self.portal_pub = rospy.Publisher('/fleet/portal_events', String, queue_size=2, latch=True)
        self.status_pub = rospy.Publisher('/fleet/execution_status', String, queue_size=2, latch=True)
        self.portal_viz_pub = rospy.Publisher('/fleet/portal_markers', MarkerArray, queue_size=1, latch=True)
        for sid in range(self.ships):
            rospy.Subscriber('/fleet/ship_{}/global_path'.format(sid), Path, self.path_cb, sid)
            rospy.Subscriber('/fleet/ship_{}/arrived'.format(sid), Bool, self.arrived_cb, sid)
        self.last = rospy.Time.now()
        self.publish_portals()
        rospy.Timer(rospy.Duration(0.1), self.step)

    def publish_portals(self):
        arr = MarkerArray()
        for i, portal in enumerate(self.portals):
            ex, ey = portal['river_entry_xy']; gx, gy = portal['river_exit_xy']
            # Show the physical extent represented by the separate localmap3
            # RViz window.  This is a geographic/visual calibration aid, not a
            # second occupancy map drawn over the whole-river map.
            bbox = portal.get('river_localmap_bbox_xy')
            if bbox and len(bbox) == 4:
                west, south, east, north = [float(v) for v in bbox]
                region = Marker(); region.header.frame_id = self.frame; region.header.stamp = rospy.Time.now(); region.ns = 'localmap3_extent'; region.id = i
                region.type = Marker.LINE_STRIP; region.action = Marker.ADD; region.pose.orientation.w = 1.0; region.scale.x = 14
                region.color.r, region.color.g, region.color.b, region.color.a = 1.0, .15, .15, .95
                region.points = [Point(west, south, 7), Point(east, south, 7), Point(east, north, 7), Point(west, north, 7), Point(west, south, 7)]
                arr.markers.append(region)
                region_label = Marker(); region_label.header.frame_id = self.frame; region_label.header.stamp = rospy.Time.now(); region_label.ns = 'localmap3_extent_label'; region_label.id = i
                region_label.type = Marker.TEXT_VIEW_FACING; region_label.action = Marker.ADD; region_label.pose.position = Point((west + east) / 2, north + 70, 18); region_label.pose.orientation.w = 1.0
                region_label.scale.z = 48; region_label.color.r, region_label.color.g, region_label.color.b, region_label.color.a = 1.0, .2, .2, 1.0
                region_label.text = 'localmap3 geographic extent'; arr.markers.append(region_label)
            # No large X gate: it obscures the dense task-allocation view.
            label = Marker(); label.header.frame_id = self.frame; label.header.stamp = rospy.Time.now(); label.ns = 'portal_labels'; label.id = i
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD; label.pose.position = Point(ex, ey + 110, 20); label.pose.orientation.w = 1.0
            label.scale.z = 58; label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = 'PORTAL: {}\nlocalmap3 entry'.format(portal['id']); arr.markers.append(label)
            link = Marker(); link.header.frame_id = self.frame; link.header.stamp = rospy.Time.now(); link.ns = 'portal_links'; link.id = i
            link.type = Marker.LINE_STRIP; link.action = Marker.ADD; link.pose.orientation.w = 1.0; link.scale.x = 10
            link.color.r, link.color.g, link.color.b, link.color.a = 1.0, .55, .0, .85; link.points = [Point(ex, ey, 8), Point(gx, gy, 8)]; arr.markers.append(link)
        self.portal_viz_pub.publish(arr)

    def arrived_cb(self, msg, sid):
        if msg.data:
            self.local_done.add(sid)
            if sid in self.local_mode:
                self.local_mode.remove(sid)
                self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'state': 'global_route_resumed'})))

    def _anchor_sequence(self, points, x, y):
        entry = min(self.anchors, key=lambda a: math.hypot(x-a['river_xy'][0], y-a['river_xy'][1]))
        if math.hypot(x-entry['river_xy'][0], y-entry['river_xy'][1]) > 105: return None
        start = min(range(len(points)), key=lambda i: math.hypot(x-points[i][0], y-points[i][1]))
        result = [entry]; last = entry['id']
        for p in points[start+1:]:
            candidates = [a for a in self.anchors if a['id'] != last and math.hypot(p[0]-a['river_xy'][0], p[1]-a['river_xy'][1]) <= 105]
            if candidates: result.append(candidates[0]); last = candidates[0]['id']
        return result if len(result) > 1 else None

    @staticmethod
    def _progress_at(points, target):
        """Arc length of the route point closest to a configured portal pose."""
        total = 0.0; best = (float('inf'), 0.0)
        for a, b in zip(points, points[1:]):
            length = math.hypot(b[0]-a[0], b[1]-a[1])
            if length:
                t = max(0.0, min(1.0, ((target[0]-a[0])*(b[0]-a[0]) + (target[1]-a[1])*(b[1]-a[1])) / (length*length)))
                px, py = a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1])
                best = min(best, (math.hypot(target[0]-px, target[1]-py), total+t*length))
            total += length
        return best[1]

    def path_cb(self, path, sid):
        points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        if len(points) < 2:
            return
        self.paths[sid] = points
        self.progress[sid] = 0.0
        self.traces[sid] = [points[0]]

    def _position(self, points, distance):
        remaining = distance
        for a, b in zip(points, points[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length and remaining <= length:
                ratio = remaining / length
                return a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])
            remaining -= length
        return points[-1]

    def step(self, _):
        now = rospy.Time.now(); dt = max(0.001, (now - self.last).to_sec()); self.last = now
        # Do not use DELETEALL here: RViz shares the marker manager among
        # MarkerArray topics, so it would erase the static map/road/task layer.
        arr = MarkerArray()
        for sid, points in self.paths.items():
            total = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))
            if total <= 0: continue
            if sid not in self.local_mode:
                self.progress[sid] = min(self.progress[sid] + self.speed * (1 + 0.06 * sid) * dt, total)
                if self.progress[sid] >= total and sid not in self.completed:
                    self.completed.add(sid)
                    self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'state': 'global_route_completed'})))
            x, y = self._position(points, self.progress[sid])
            if self.anchors and sid not in self.pending_local:
                sequence = self._anchor_sequence(points, x, y)
                if sequence:
                    entry, exit_ = sequence[0], sequence[-1]; lane = 1 if exit_['local_xy'][0] >= entry['local_xy'][0] else -1
                    self.pending_local[sid] = self._progress_at(points, exit_['river_xy'])
                    self.portal_pub.publish(String(data=json.dumps({'ship_id': sid, 'entry_anchor': entry['id'], 'exit_anchor': exit_['id'], 'local_waypoints': [a['local_xy'] for a in sequence], 'lane_direction': lane})))
                    self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'state': 'local_dispatched', 'entry': entry['id'], 'exit': exit_['id']})))
            for portal in ([] if self.anchors else self.portals):
                key = (sid, portal.get('id'))
                if key in self.triggered_portals:
                    continue
                ex, ey = portal['river_entry_xy']
                hit = math.hypot(x - ex, y - ey) <= float(portal.get('trigger_radius', 100.0))
                # Exact localmap3 extent test: any allocated route entering the
                # calibrated geographic rectangle triggers the matching travel
                # direction, even if it uses a different road node on its edge.
                bbox = portal.get('river_localmap_bbox_xy')
                if bbox:
                    west, south, east, north = [float(v) for v in bbox]
                    prev = self.traces.get(sid, [(x, y)])[-1]
                    direction = 1 if x >= prev[0] else -1
                    hit = hit or (west <= x <= east and south <= y <= north and
                                  direction == int(portal.get('lane_direction', 1)))
                if hit:
                    self.portal_pub.publish(String(data=json.dumps({'ship_id': sid, 'portal_id': portal['id']})))
                    self.triggered_portals.add(key)
                    self.local_mode.add(sid)
                    self.portal_exit_progress[sid] = self._progress_at(points, portal['river_exit_xy'])
                    self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'portal_id': portal['id'], 'state': 'portal_entered_local_planning'})))
            if sid in self.pending_local and sid not in self.local_mode and self.progress[sid] >= self.pending_local[sid]:
                self.local_mode.add(sid)
                self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'state': 'waiting_at_local_exit'})))
                if sid in self.local_done:
                    self.local_mode.remove(sid)
                    self.status_pub.publish(String(data=json.dumps({'ship_id': sid, 'state': 'global_route_resumed'})))
            trace = self.traces.setdefault(sid, []); trace.append((x, y)); trace[:] = trace[-240:]
            hue = (sid * .31) % 1.0
            ship = Marker(); ship.header.frame_id = self.frame; ship.header.stamp = now; ship.ns = 'moving_ships'; ship.id = sid
            ship.type = Marker.SPHERE; ship.action = Marker.ADD; ship.pose.position = Point(x, y, 5); ship.pose.orientation.w = 1.0
            ship.scale.x = ship.scale.y = 80; ship.scale.z = 28; ship.color.r = hue; ship.color.g = .95; ship.color.b = 1 - hue; ship.color.a = 1.0; arr.markers.append(ship)
            label = Marker(); label.header.frame_id = self.frame; label.header.stamp = now; label.ns = 'moving_labels'; label.id = sid
            label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD; label.pose.position = Point(x, y, 14); label.pose.orientation.w = 1.0
            label.scale.z = 48; label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = 'USV-{}\n{}'.format(sid, 'LOCAL PLANNING' if sid in self.local_mode else 'GLOBAL ROUTE'); arr.markers.append(label)
            line = Marker(); line.header.frame_id = self.frame; line.header.stamp = now; line.ns = 'ship_traces'; line.id = sid
            line.type = Marker.LINE_STRIP; line.action = Marker.ADD; line.pose.orientation.w = 1.0; line.scale.x = 9.0
            line.color.r = hue; line.color.g = .95; line.color.b = 1 - hue; line.color.a = .9
            line.points = [Point(px, py, 3) for px, py in trace]; arr.markers.append(line)
        self.pub.publish(arr)
        if self.paths and len(self.completed) == len(self.paths) and not self.all_complete_announced:
            self.all_complete_announced = True
            self.status_pub.publish(String(data=json.dumps({'state': 'fleet_completed'})))


if __name__ == '__main__':
    FleetMotionSim(); rospy.spin()
