#!/usr/bin/env python3
"""从 road_network.json 生成路网图"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from visualize_road_net import visualize_road_network_on_map

BASE = os.path.dirname(os.path.abspath(__file__))
JSON = f'{BASE}/../output/road_network.json'
MAP  = f'{BASE}/../../../data/maps/binary_map_scaled.png'
OUT  = f'{BASE}/../output/road_network_on_map.png'

rn = load_road_network(JSON, f'{BASE}/../config/ports.yaml')
visualize_road_network_on_map(rn, MAP, OUT)
