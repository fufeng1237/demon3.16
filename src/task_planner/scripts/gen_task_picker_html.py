#!/usr/bin/env python3
"""生成交互式任务选择器HTML"""
import sys,os,json,base64
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from road_network import load_road_network
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.dirname(os.path.abspath(__file__))
rn=load_road_network(f'{BASE}/../output/road_network.json',f'{BASE}/../config/ports.yaml')
img=np.array(Image.open('/root/demon3.16/data/maps/binary_map_scaled.png'))
if img.ndim==3: img=img[:,:,0]
h,w=img.shape; ps=2.0; wh=h*ps; ww=w*ps
bg=np.flipud(img); water=bg>127; wr,wc=np.where(water); m=50

# 渲染地图
fig,ax=plt.subplots(figsize=(20,9))
ax.imshow(bg,extent=[0,ww,0,wh],origin='lower',cmap='gray')
for e in rn.edges:
    n1,n2=rn.nodes[e.from_id],rn.nodes[e.to_id]
    ax.plot([n1.x,n2.x],[wh-n1.y,wh-n2.y],'cyan',lw=0.5,alpha=0.4,zorder=1)
for n in rn.nodes.values():
    if n.is_port:
        ax.scatter(n.x,wh-n.y,c='red',s=60,marker='s',edgecolors='white',lw=1,zorder=3)
        ax.annotate(n.port_name,(n.x,wh-n.y+20),fontsize=7,color='white',ha='center',zorder=5)
ax.set_xlim(max(0,wc.min()-m)*ps,min(w,wc.max()+m)*ps)
ax.set_ylim(max(0,wr.min()-m)*ps,min(h,wr.max()+m)*ps)
ax.set_aspect('equal')
plt.tight_layout()
buf=BytesIO(); plt.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor='#111'); plt.close()
b64=base64.b64encode(buf.getvalue()).decode()

# 生成HTML
ports_js=json.dumps([{'n':n.port_name,'x':n.x,'y':n.y} for n in rn.nodes.values() if n.is_port])
gas_js=json.dumps([{'n':n.port_name,'x':n.x,'y':n.y} for n in rn.nodes.values() if n.is_gas_station])

html=f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Task Picker</title>
<style>
body{{margin:0;background:#111;color:#fff;font-family:monospace;overflow:hidden}}
#top{{position:fixed;top:8px;left:8px;z-index:10;background:#222;padding:8px 12px;border-radius:6px;font-size:13px}}
#top span{{color:#0f0;font-weight:bold}}
#side{{position:fixed;top:8px;right:8px;z-index:10;background:#222;padding:8px;border-radius:6px;max-height:85vh;overflow-y:auto;min-width:260px;font-size:12px}}
#status{{position:fixed;bottom:8px;left:8px;z-index:10;background:#222;padding:6px 12px;border-radius:6px;font-size:13px}}
button{{margin:2px 4px;padding:5px 10px;cursor:pointer;border:none;border-radius:4px;font-size:12px}}
.btn-save{{background:#0a0;color:#fff}}.btn-undo{{background:#a40;color:#fff}}.btn-clear{{background:#444;color:#fff}}
.pu{{color:#0f0}}.de{{color:#f44}}.tsk{{color:#ff0}}
</style></head><body>
<div id="top"><b>Task Picker</b><br><span id="cnt">Tasks: 0</span><br>
<small class="pu">Click: Pickup</small> → <small class="de">Delivery</small><br>
<small>Right-click: undo point</small></div>
<div id="side"><b>Tasks:</b><br><span id="lst" style="color:#888">None</span></div>
<div id="status">Click map to place Pickup</div>
<canvas id="c"></canvas>
<script>
const PORTS={ports_js};
const GAS={gas_js};
const WH={wh:.0f};const WW={ww:.0f};const PS={ps};
let tasks=[],cur=null;
const c=document.getElementById('c'),ctx=c.getContext('2d');
const bgImg=new Image();
bgImg.src='data:image/png;base64,{b64}';
let bgLoaded=false;
bgImg.onload=()=>{{bgLoaded=true;resize();draw();}};

function resize(){{
    c.width=window.innerWidth;c.height=window.innerHeight;
    draw();
}}
window.addEventListener('resize',resize);

function w2s(wx,wy){{return{{x:wx/WW*c.width,y:(WH-wy)/WH*c.height}};}}
function s2w(sx,sy){{return{{x:sx/c.width*WW,y:WH-sy/c.height*WH}};}}

function snap(wx,wy){{return{{x:Math.round(wx/10)*10,y:Math.round(wy/10)*10}};}}

c.addEventListener('click',e=>{{
    if(!bgLoaded)return;
    const r=c.getBoundingClientRect();
    const w=s2w(e.clientX-r.left,e.clientY-r.top);
    const s=snap(w.x,w.y);
    if(!cur){{
        cur=s;
        document.getElementById('status').innerHTML=`<span class="pu">Pickup</span> at (${{s.x.toFixed(0)}},${{s.y.toFixed(0)}})m — now click <span class="de">Delivery</span>`;
    }}else{{
        tasks.push({{px:cur.x,py:cur.y,dx:s.x,dy:s.y}});
        cur=null;
        document.getElementById('status').innerHTML=`Task ${{tasks.length-1}} done (${{tasks.length}} total)`;
        updList();
    }}
    draw();
}});

c.addEventListener('contextmenu',e=>{{e.preventDefault();if(cur){{cur=null;draw();}}}});

function undo(){{if(tasks.length){{tasks.pop();updList();draw();}}}}
function clearAll(){{tasks=[];cur=null;updList();draw();}}
function save(){{
    let t='# Tasks (world coords)\\n';
    tasks.forEach((tk,i)=>{{t+=`Task ${{i}}: pickup=(${{tk.px.toFixed(0)}},${{tk.py.toFixed(0)}}) delivery=(${{tk.dx.toFixed(0)}},${{tk.dy.toFixed(0)}})\\n`;}});
    const b=new Blob([t],{{type:'text/plain'}});
    const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='tasks.txt';a.click();
}};

function updList(){{
    document.getElementById('cnt').textContent=`Tasks: ${{tasks.length}}`;
    if(!tasks.length){{document.getElementById('lst').innerHTML='<span style=color:#888>None</span>';return;}}
    let h='';
    tasks.forEach((tk,i)=>{{h+=`<span class=tsk>T${{i}}</span> <span class=pu>P</span>(${{tk.px.toFixed(0)}},${{tk.py.toFixed(0)}})→<span class=de>D</span>(${{tk.dx.toFixed(0)}},${{tk.dy.toFixed(0)}})<br>`;}});
    document.getElementById('lst').innerHTML=h;
}}

function draw(){{
    ctx.clearRect(0,0,c.width,c.height);
    // Background
    if(bgLoaded)ctx.drawImage(bgImg,0,0,c.width,c.height);
    // Tasks
    tasks.forEach((tk,i)=>{{
        const p1=w2s(tk.px,tk.py),p2=w2s(tk.dx,tk.dy);
        ctx.strokeStyle='rgba(255,255,0,0.6)';ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();
        // Pickup
        ctx.fillStyle='#0f0';ctx.beginPath();ctx.arc(p1.x,p1.y,8,0,7);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
        // Delivery
        ctx.fillStyle='#f44';const s=8;ctx.beginPath();
        ctx.moveTo(p2.x-s,p2.y-s);ctx.lineTo(p2.x+s,p2.y+s);
        ctx.moveTo(p2.x+s,p2.y-s);ctx.lineTo(p2.x-s,p2.y+s);
        ctx.lineWidth=3;ctx.strokeStyle='#fff';ctx.stroke();
        // Label
        ctx.fillStyle='#ff0';ctx.font='bold 11px monospace';
        ctx.fillText('T'+i,(p1.x+p2.x)/2,(p1.y+p2.y)/2-10);
    }});
    // Current pickup
    if(cur){{
        const p=w2s(cur.x,cur.y);
        ctx.fillStyle='#0f0';ctx.beginPath();ctx.arc(p.x,p.y,12,0,7);ctx.fill();
        ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke();
        ctx.fillStyle='#fff';ctx.font='bold 14px monospace';ctx.fillText('PICKUP',p.x+16,p.y+5);
    }}
}}
</script>
<div style="position:fixed;bottom:50px;left:8px;z-index:10">
<button class="btn-save" onclick="save()">💾 Save</button>
<button class="btn-undo" onclick="undo()">↩ Undo</button>
<button class="btn-clear" onclick="clearAll()">🗑 Clear</button>
</div>
</body></html>'''

out=f'{BASE}/../output/task_picker.html'
with open(out,'w') as f: f.write(html)
print(f'Generated: {out}')
print(f'Open in browser: file://{os.path.abspath(out)}')
