#!/usr/bin/env python3
"""Reference-path DWA controller for one localmap3 vessel.

The AIS smoothed path is the global reference.  Dynamic-window candidates are
scored for path tracking, forward progress and map collision clearance.
"""
import math
import rospy
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid


class AISDWATracker:
    def __init__(self):
        rospy.init_node('ais_dwa_tracker')
        sid = int(rospy.get_param('~ship_id', 0))
        self.ns = rospy.get_param('~local_ns', '/fleet/local/ship_{}'.format(sid)).rstrip('/')
        self.path = []; self.pose = None; self.yaw = 0.0; self.v = 0.0; self.goal = None; self.grid = None
        self.max_v = float(rospy.get_param('~max_v', 2.0)); self.max_w = float(rospy.get_param('~max_w', 1.0))
        self.pub = rospy.Publisher(self.ns + '/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber(self.ns + '/ais_smoothed_path', Path, self.path_cb)
        rospy.Subscriber(self.ns + '/odom', Odometry, self.odom_cb)
        rospy.Subscriber(self.ns + '/initialpose', PoseWithCovarianceStamped, self.initial_cb)
        rospy.Subscriber(self.ns + '/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/map', OccupancyGrid, self.map_cb)
        rospy.Timer(rospy.Duration(.1), self.step)

    def path_cb(self, msg): self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
    def map_cb(self, msg): self.grid = msg
    def initial_cb(self, msg): self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y); self.yaw = 0.0; self.v = 0.0
    def goal_cb(self, msg): self.goal = (msg.pose.position.x, msg.pose.position.y)
    def odom_cb(self, msg):
        p = msg.pose.pose; self.pose = (p.position.x, p.position.y); self.v = msg.twist.twist.linear.x
        self.yaw = math.atan2(2*(p.orientation.w*p.orientation.z), 1-2*p.orientation.z*p.orientation.z)

    def blocked(self, x, y):
        if not self.grid: return False
        info = self.grid.info; ix = int((x-info.origin.position.x)/info.resolution); iy = int((y-info.origin.position.y)/info.resolution)
        return ix < 0 or iy < 0 or ix >= info.width or iy >= info.height or self.grid.data[iy*info.width+ix] > 50

    def step(self, _):
        if not self.pose or len(self.path) < 2 or not self.goal: return
        x, y = self.pose
        if math.hypot(x-self.goal[0], y-self.goal[1]) < .7:
            self.pub.publish(Twist()); return
        nearest = min(range(len(self.path)), key=lambda i: math.hypot(x-self.path[i][0], y-self.path[i][1]))
        target = self.path[min(nearest + 5, len(self.path)-1)]
        best = None
        # Dynamic window: reachable v/w in the coming control interval.
        for v in [max(0., self.v-.25)+.25*i for i in range(9)]:
            if v > self.max_v: continue
            for w in [-self.max_w+.25*i for i in range(9)]:
                px, py, yaw = x, y, self.yaw; safe = True
                for _ in range(12):
                    yaw += w*.1; px += v*math.cos(yaw)*.1; py += v*math.sin(yaw)*.1
                    if self.blocked(px, py): safe = False; break
                if not safe: continue
                err = math.hypot(px-target[0], py-target[1])
                heading = abs(math.atan2(target[1]-py, target[0]-px)-yaw); heading = min(heading, 2*math.pi-heading)
                score = 3.0*err + .45*heading - .35*v
                if best is None or score < best[0]: best = (score, v, w)
        cmd = Twist()
        if best: cmd.linear.x, cmd.angular.z = best[1], best[2]
        self.pub.publish(cmd)


if __name__ == '__main__': AISDWATracker(); rospy.spin()
