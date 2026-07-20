#!/usr/bin/env python3
"""Idealised local-planner substitute for an end-to-end RViz demonstration.

It follows an AIS global path exactly at constant speed.  It has no obstacle
avoidance or dynamics model and must be replaced by the collaborator's local
planner for experimental claims.
"""
import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
import math


class MockLocalFollower:
    def __init__(self):
        rospy.init_node('mock_local_follower')
        self.ship_id = int(rospy.get_param('~ship_id', 0))
        self.speed = float(rospy.get_param('~speed_mps', 3.0))
        self.rate = float(rospy.get_param('~rate_hz', 10.0))
        self.path, self.segment, self.distance = None, 0, 0.0
        self.pose_pub = rospy.Publisher('/fleet/ais_demo/mock_pose', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/fleet/ais_demo/mock_ship_marker', Marker, queue_size=1)
        self.arrived_pub = rospy.Publisher('/fleet/ship_{}/arrived'.format(self.ship_id), Bool, queue_size=1)
        self.status_pub = rospy.Publisher('/fleet/ais_demo/mock_status', String, queue_size=1, latch=True)
        rospy.Subscriber('/fleet/ais_demo/global_path', Path, self.path_cb)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.step)
        rospy.loginfo('Mock local follower ready: ship=%d speed=%.1fm/s', self.ship_id, self.speed)

    def path_cb(self, path):
        if len(path.poses) < 2:
            self.status_pub.publish(String(data='rejected_path: fewer than two poses'))
            return
        self.path, self.segment, self.distance = path, 0, 0.0
        self.status_pub.publish(String(data='following_path ship={} poses={}'.format(self.ship_id, len(path.poses))))

    def step(self, event):
        if self.path is None:
            return
        self.distance += self.speed / self.rate
        poses = self.path.poses
        while self.segment < len(poses) - 1:
            a, b = poses[self.segment].pose.position, poses[self.segment + 1].pose.position
            length = math.hypot(b.x - a.x, b.y - a.y)
            if self.distance <= length or length < 1e-6:
                ratio = 1.0 if length < 1e-6 else self.distance / length
                self.publish_pose(a.x + ratio * (b.x - a.x), a.y + ratio * (b.y - a.y), self.path.header.frame_id)
                return
            self.distance -= length; self.segment += 1
        end = poses[-1].pose.position; self.publish_pose(end.x, end.y, self.path.header.frame_id)
        self.arrived_pub.publish(Bool(data=True))
        self.status_pub.publish(String(data='arrived ship={} (mock follower)'.format(self.ship_id)))
        self.path = None

    def publish_pose(self, x, y, frame):
        pose = PoseStamped(); pose.header.frame_id = frame or 'map'; pose.header.stamp = rospy.Time.now()
        pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.w = x, y, 1.0; self.pose_pub.publish(pose)
        marker = Marker(); marker.header = pose.header; marker.ns = 'mock_local_ship'; marker.id = self.ship_id
        marker.type = Marker.SPHERE; marker.action = Marker.ADD; marker.pose = pose.pose
        marker.scale.x = marker.scale.y = 3.0; marker.scale.z = 1.5; marker.color.r = 1.0; marker.color.g = .5; marker.color.a = 1.0
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    MockLocalFollower(); rospy.spin()
