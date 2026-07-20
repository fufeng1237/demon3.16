#!/usr/bin/env python3
"""Minimal unicycle simulator for ROS move_base + DWA visual demonstrations."""
import math
import rospy
import tf2_ros
from geometry_msgs.msg import Twist, TransformStamped, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


class DWASim:
    def __init__(self):
        rospy.init_node('dwa_kinematic_sim')
        self.ship_id = int(rospy.get_param('~ship_id', 3))
        self.ns = rospy.get_param('~local_ns', '/fleet/local/ship_{}'.format(self.ship_id)).rstrip('/')
        self.x, self.y, self.yaw = 14.25, 25.25, 0.0
        self.v, self.w = 0.0, 0.0; self.goal = None; self.arrived = False; self.active = False
        self.base = rospy.get_param('~base_frame', 'base_link'); self.frame = 'map'
        self.odom_pub = rospy.Publisher(self.ns + '/odom', Odometry, queue_size=10)
        self.path_pub = rospy.Publisher(self.ns + '/dwa_trace', Path, queue_size=1, latch=True)
        self.arrived_pub = rospy.Publisher(self.ns + '/sim_arrived', Bool, queue_size=1)
        self.status_pub = rospy.Publisher(self.ns + '/dwa_status', String, queue_size=1, latch=True)
        self.viz_pub = rospy.Publisher('/fleet/local/vessel_markers', MarkerArray, queue_size=1)
        self.trace = Path(); self.trace.header.frame_id = self.frame
        self.tf = tf2_ros.TransformBroadcaster()
        self.last_viz = rospy.Time(0)
        rospy.Subscriber(self.ns + '/cmd_vel', Twist, self.cmd_cb)
        rospy.Subscriber(self.ns + '/initialpose', PoseWithCovarianceStamped, self.initial_cb)
        rospy.Subscriber(self.ns + '/goal', PoseStamped, self.goal_cb)
        self.last = rospy.Time.now(); rospy.Timer(rospy.Duration(.05), self.step)

    def cmd_cb(self, msg): self.v, self.w = msg.linear.x, msg.angular.z
    def initial_cb(self, msg):
        self.x, self.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.yaw = 0.0; self.trace.poses = []; self.arrived = False; self.active = True
    def goal_cb(self, msg): self.goal = (msg.pose.position.x, msg.pose.position.y); self.arrived = False

    def step(self, _):
        now = rospy.Time.now(); dt = max(0.001, (now - self.last).to_sec()); self.last = now
        if not self.active:
            return
        self.x += self.v * math.cos(self.yaw) * dt; self.y += self.v * math.sin(self.yaw) * dt; self.yaw += self.w * dt
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        tf = TransformStamped(); tf.header.stamp = now; tf.header.frame_id = self.frame; tf.child_frame_id = self.base
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y; tf.transform.rotation.z, tf.transform.rotation.w = qz, qw; self.tf.sendTransform(tf)
        odom = Odometry(); odom.header = tf.header; odom.child_frame_id = self.base; odom.pose.pose.position.x = self.x; odom.pose.pose.position.y = self.y; odom.pose.pose.orientation.z = qz; odom.pose.pose.orientation.w = qw; odom.twist.twist.linear.x = self.v; odom.twist.twist.angular.z = self.w; self.odom_pub.publish(odom)
        pose = PoseStamped(); pose.header = tf.header; pose.pose = odom.pose.pose; self.trace.poses.append(pose)
        if len(self.trace.poses) > 600: self.trace.poses = self.trace.poses[-600:]
        self.trace.header.stamp = now; self.path_pub.publish(self.trace)
        # RViz receives markers from every potential local vessel.  Limit the
        # visual stream while retaining the 20 Hz control/odometry loop.
        if (now - self.last_viz).to_sec() >= .2:
            self.publish_vessel(now, qz, qw); self.last_viz = now
        if self.goal and not self.arrived and math.hypot(self.x-self.goal[0], self.y-self.goal[1]) < .75:
            self.arrived = True; self.arrived_pub.publish(Bool(data=True)); self.status_pub.publish(String(data='dwa_arrived'))

    def publish_vessel(self, now, qz, qw):
        arr = MarkerArray()
        hull = Marker(); hull.header.frame_id = self.frame; hull.header.stamp = now; hull.ns = 'usv_hull'; hull.id = self.ship_id
        hull.type = Marker.ARROW; hull.action = Marker.ADD; hull.pose.position.x = self.x; hull.pose.position.y = self.y; hull.pose.position.z = 2.0
        hull.pose.orientation.z, hull.pose.orientation.w = qz, qw; hull.scale.x, hull.scale.y, hull.scale.z = 4.5, 1.4, 1.4
        hull.color.r, hull.color.g, hull.color.b, hull.color.a = .0, .85, 1.0, 1.0; arr.markers.append(hull)
        wake = Marker(); wake.header.frame_id = self.frame; wake.header.stamp = now; wake.ns = 'usv_wake'; wake.id = self.ship_id
        wake.type = Marker.LINE_STRIP; wake.action = Marker.ADD; wake.pose.orientation.w = 1.0; wake.scale.x = .32
        wake.color.r, wake.color.g, wake.color.b, wake.color.a = 1.0, .82, .0, 1.0
        wake.points = [p.pose.position for p in self.trace.poses[-300:]]; arr.markers.append(wake)
        label = Marker(); label.header.frame_id = self.frame; label.header.stamp = now; label.ns = 'usv_label'; label.id = self.ship_id
        label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD; label.pose.position.x = self.x; label.pose.position.y = self.y; label.pose.position.z = 4.5; label.pose.orientation.w = 1.0
        label.scale.z = 1.6; label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = 'USV-{} | DWA v={:.2f}'.format(self.ship_id, self.v); arr.markers.append(label)
        self.viz_pub.publish(arr)

if __name__ == '__main__': DWASim(); rospy.spin()
