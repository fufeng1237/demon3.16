#!/usr/bin/env python3
"""Aggregate independent AIS local references into one RViz MarkerArray."""
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

COLORS = [(1,.2,.2), (.2,1,.2), (.2,.6,1), (1,.8,.1), (1,.2,1), (.1,1,1), (1,.5,.1), (.8,.8,.8)]
class LocalPathViz:
    def __init__(self):
        rospy.init_node('local_path_viz'); self.paths = {}
        self.pub=rospy.Publisher('/fleet/local/ais_reference_markers', MarkerArray,queue_size=1)
        for sid in range(int(rospy.get_param('~ships',8))):
            rospy.Subscriber('/fleet/local/ship_{}/global_path'.format(sid),Path,self.cb,sid)
    def cb(self,msg,sid): self.paths[sid]=msg; self.publish()
    def publish(self):
        arr=MarkerArray()
        for sid,path in self.paths.items():
            r,g,b=COLORS[sid%len(COLORS)]; m=Marker(); m.header.frame_id='map';m.header.stamp=rospy.Time.now();m.ns='ais_reference';m.id=sid;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.pose.orientation.w=1;m.scale.x=.42;m.color.r=r;m.color.g=g;m.color.b=b;m.color.a=1
            m.points=[Point(p.pose.position.x,p.pose.position.y,1.3) for p in path.poses];arr.markers.append(m)
        self.pub.publish(arr)
if __name__=='__main__': LocalPathViz(); rospy.spin()
