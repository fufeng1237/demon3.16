#!/usr/bin/env python3
"""Publish the task-planner's full-river binary image as a ROS map."""
import os
import numpy as np
from PIL import Image
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def main():
    rospy.init_node('whole_river_map')
    default = '/root/demon3.16/data/maps/binary_map_scaled.png'
    image_path = rospy.get_param('~image', default)
    frame = rospy.get_param('~frame_id', 'river_map')
    resolution = float(rospy.get_param('~resolution', 2.0))
    image = np.asarray(Image.open(image_path).convert('L'))
    height, width = image.shape
    grid = OccupancyGrid()
    grid.header.frame_id = frame
    grid.info.resolution = resolution
    grid.info.width, grid.info.height = width, height
    grid.info.origin.orientation.w = 1.0
    # PNG starts at the top; OccupancyGrid data starts at the lower-left.
    grid.data = np.where(np.flipud(image) > 127, 0, 100).astype(np.int8).ravel().tolist()
    publisher = rospy.Publisher('/fleet/global_map', OccupancyGrid, queue_size=1, latch=True)
    visual_pub = rospy.Publisher('/fleet/river_background', MarkerArray, queue_size=1, latch=True)
    grid.header.stamp = rospy.Time.now()
    publisher.publish(grid)
    # A sampled water-surface marker is clearer than a large black occupancy
    # texture in RViz and uses the exact same image-to-world conversion as the
    # road network: world_y = image_height - image_row.
    step = int(rospy.get_param('~visual_step', 6))
    yy, xx = np.where(image[::step, ::step] > 127)
    water = Marker()
    water.header.frame_id = frame
    water.header.stamp = rospy.Time.now()
    water.ns = 'river_water'
    water.id = 0
    water.type = Marker.POINTS
    water.action = Marker.ADD
    water.pose.orientation.w = 1.0
    water.scale.x = water.scale.y = resolution * step * 1.15
    water.color.r, water.color.g, water.color.b, water.color.a = 0.02, 0.35, 0.70, 0.88
    water.points = [Point(float(x * step), float(height - 1 - y * step), -0.2)
                    for y, x in zip(yy, xx)]
    visual_pub.publish(MarkerArray(markers=[water]))
    rospy.loginfo('Whole river map published: %s (%dx%d, water samples=%d)', image_path, width, height, len(water.points))
    rospy.spin()


if __name__ == '__main__':
    main()
