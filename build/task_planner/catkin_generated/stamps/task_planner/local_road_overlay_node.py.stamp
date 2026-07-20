#!/usr/bin/env python3
"""Project the full-river road-network subset into localmap3 coordinates."""
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from road_network import load_road_network
class Overlay:
 def __init__(self):
  rospy.init_node('local_road_overlay'); rn=load_road_network('/root/demon3.16/src/task_planner/output/road_network.json','/root/demon3.16/src/task_planner/config/ports.txt')
  w,s,e,n=5959.4667,1882.3259,6537.8401,2184.9253; m=Marker();m.header.frame_id='map';m.ns='projected_global_road';m.id=0;m.type=Marker.LINE_LIST;m.action=Marker.ADD;m.pose.orientation.w=1;m.scale.x=.16;m.color.r=.1;m.color.g=.9;m.color.b=1;m.color.a=.8
  def p(node): return Point((node.x-w)/(e-w)*76,(3252-node.y-s)/(n-s)*39,.8)
  for ed in rn.edges:
   a,b=rn.nodes[ed.from_id],rn.nodes[ed.to_id]
   if w<=a.x<=e and s<=3252-a.y<=n and w<=b.x<=e and s<=3252-b.y<=n:m.points += [p(a),p(b)]
  pub=rospy.Publisher('/fleet/local/projected_road',Marker,queue_size=1,latch=True);m.header.stamp=rospy.Time.now();pub.publish(m)
if __name__=='__main__': Overlay();rospy.spin()
