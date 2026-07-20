#!/usr/bin/env python3
"""Generate a browser-based pickup/delivery selector for the current map.

The selector exports ``tasks.txt`` in the C++ generator's native image-pixel
coordinate system: x grows rightward and y grows downward from the top edge.
"""
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from scripts.core.road_network import load_road_network


PACKAGE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PACKAGE_DIR.parents[1]
MAP_PATH = WORKSPACE_DIR / "data" / "maps" / "binary_map_scaled.png"
NETWORK_PATH = PACKAGE_DIR / "output" / "road_network.json"
PORTS_PATH = PACKAGE_DIR / "config" / "ports.yaml"
OUTPUT_PATH = PACKAGE_DIR / "output" / "task_picker.html"


def png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    image = Image.open(MAP_PATH).convert("RGB")
    width, height = image.size
    network = load_road_network(str(NETWORK_PATH), str(PORTS_PATH))

    # RoadNetwork coordinates are in world units; the task parser expects the
    # original map pixels, therefore convert ports back by the shared scale.
    pixel_scale = 2.0
    ports = [
        {"name": node.port_name, "x": node.x / pixel_scale, "y": node.y / pixel_scale}
        for node in network.nodes.values() if node.is_port
    ]
    gas_stations = [
        {"name": node.port_name, "x": node.x / pixel_scale, "y": node.y / pixel_scale}
        for node in network.nodes.values() if node.is_gas_station
    ]

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Task Picker</title>
<style>
body {{ margin:0; overflow:hidden; background:#111; color:#fff; font:13px monospace; }}
#top,#side,#status,#cursor {{ position:fixed; z-index:10; background:#222d; border-radius:6px; padding:8px 12px; }}
#top {{ top:8px; left:8px; }} #side {{ top:8px; right:8px; max-height:84vh; overflow-y:auto; min-width:285px; }}
#status {{ bottom:8px; left:8px; }} #cursor {{ bottom:8px; right:8px; color:#aaa; }}
button {{ margin:2px; padding:5px 10px; border:0; border-radius:4px; cursor:pointer; color:white; }}
.save {{ background:#087f23; }} .undo {{ background:#a64518; }} .clear {{ background:#555; }}
.pickup {{ color:#32ff6a; }} .delivery {{ color:#ff5959; }} .task {{ color:#ffe45e; }}
canvas {{ display:block; }}
</style></head><body>
<div id="top"><b>Task Picker — 新地图</b><br>原图像素坐标：{width} × {height}<br>
<span id="count">Tasks: 0</span><br><span class="pickup">左键：Pickup</span> → <span class="delivery">Delivery</span><br>
右键撤销当前点；保存后直接替换 <code>tasks.txt</code></div>
<div id="side"><b>Tasks</b><br><span id="list" style="color:#999">尚未选择</span></div>
<div id="status">在水道上点击以放置 Pickup</div><div id="cursor">px: 0, 0</div>
<canvas id="map"></canvas>
<div style="position:fixed; left:8px; bottom:48px; z-index:10">
<button class="save" onclick="saveTasks()">保存 tasks.txt</button><button class="undo" onclick="undo()">撤销</button><button class="clear" onclick="clearTasks()">清空</button></div>
<script>
const WIDTH={width}, HEIGHT={height};
const PORTS={json.dumps(ports, ensure_ascii=False)};
const GAS={json.dumps(gas_stations, ensure_ascii=False)};
const canvas=document.getElementById('map'), ctx=canvas.getContext('2d');
const image=new Image(); image.src={json.dumps(png_data_url(image))};
let scale=1, offsetX=0, offsetY=0, tasks=[], pickup=null;

function resize() {{
  canvas.width=innerWidth; canvas.height=innerHeight;
  scale=Math.min(canvas.width/WIDTH, canvas.height/HEIGHT);
  offsetX=(canvas.width-WIDTH*scale)/2; offsetY=(canvas.height-HEIGHT*scale)/2; draw();
}}
function toPixel(x,y) {{ return {{x:Math.round((x-offsetX)/scale), y:Math.round((y-offsetY)/scale)}}; }}
function toScreen(x,y) {{ return {{x:offsetX+x*scale, y:offsetY+y*scale}}; }}
function valid(p) {{ return p.x>=0 && p.x<WIDTH && p.y>=0 && p.y<HEIGHT; }}
function snap(p) {{ return {{x:Math.round(p.x/5)*5, y:Math.round(p.y/5)*5}}; }}

canvas.addEventListener('mousemove', e => {{ const r=canvas.getBoundingClientRect(), p=toPixel(e.clientX-r.left,e.clientY-r.top); document.getElementById('cursor').textContent=`px: ${{p.x}}, ${{p.y}}`; }});
canvas.addEventListener('click', e => {{
  const r=canvas.getBoundingClientRect(); let p=snap(toPixel(e.clientX-r.left,e.clientY-r.top)); if(!valid(p)) return;
  if(pickup===null) {{ pickup=p; document.getElementById('status').innerHTML=`<span class="pickup">Pickup</span> = (${{p.x}}, ${{p.y}})，请点击 Delivery`; }}
  else {{ tasks.push({{px:pickup.x,py:pickup.y,dx:p.x,dy:p.y}}); pickup=null; document.getElementById('status').textContent=`已完成任务 ${{tasks.length-1}}`; updateList(); }} draw();
}});
canvas.addEventListener('contextmenu', e => {{ e.preventDefault(); pickup=null; draw(); }});
window.addEventListener('resize', resize); image.onload=resize;

function updateList() {{
  document.getElementById('count').textContent=`Tasks: ${{tasks.length}}`;
  document.getElementById('list').innerHTML=tasks.length ? tasks.map((t,i)=>`<span class="task">T${{i}}</span> <span class="pickup">P</span>(${{t.px}},${{t.py}}) → <span class="delivery">D</span>(${{t.dx}},${{t.dy}})`).join('<br>') : '<span style="color:#999">尚未选择</span>';
}}
function undo() {{ if(pickup!==null) pickup=null; else tasks.pop(); updateList(); draw(); }}
function clearTasks() {{ tasks=[]; pickup=null; updateList(); draw(); }}
function saveTasks() {{
  let text='# Tasks — C++ pixel coords (original image, y=0 at top)\\n# Format: Task ID: pickup=(px,py) delivery=(px,py)\\n\\n';
  tasks.forEach((t,i)=>text+=`Task ${{i}}: pickup=(${{t.px}},${{t.py}}) delivery=(${{t.dx}},${{t.dy}})\\n`);
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{{type:'text/plain'}})); a.download='tasks.txt'; a.click();
}}
function marker(p, color, label, square=false) {{ const q=toScreen(p.x,p.y); ctx.fillStyle=color; ctx.strokeStyle='#fff'; ctx.lineWidth=1; if(square) {{ctx.fillRect(q.x-3,q.y-3,6,6);ctx.strokeRect(q.x-3,q.y-3,6,6);}} else {{ctx.beginPath();ctx.arc(q.x,q.y,3,0,Math.PI*2);ctx.fill();ctx.stroke();}} if(label) {{ctx.fillStyle='#fff';ctx.font='11px monospace';ctx.fillText(label,q.x+5,q.y-5);}} }}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(image,offsetX,offsetY,WIDTH*scale,HEIGHT*scale);
  PORTS.forEach(p=>marker(p,'#e22',p.name,true)); GAS.forEach(p=>marker(p,'#1cdb49',p.name));
  tasks.forEach((t,i)=>{{const a=toScreen(t.px,t.py),b=toScreen(t.dx,t.dy);ctx.strokeStyle='#ffe45e';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();marker({{x:t.px,y:t.py}},'#32ff6a','P'+i);marker({{x:t.dx,y:t.dy}},'#ff5959','D'+i);}});
  if(pickup) marker(pickup,'#32ff6a','Pickup');
}}
</script></body></html>'''
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Generated: {OUTPUT_PATH}")
    print(f"Map: {MAP_PATH} ({width}x{height})")


if __name__ == "__main__":
    main()
