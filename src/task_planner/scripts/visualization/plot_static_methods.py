#!/usr/bin/env python3
"""Plot unified static-method comparison from learning evaluator JSON."""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=json.loads(Path(a.input).read_text())['mean']; names=list(result)
    metrics=[('makespan_s','Makespan (s)'),('distance_m','Distance (m)'),('energy_kwh','Energy (kWh)')]
    fig,axes=plt.subplots(1,3,figsize=(16,4.7))
    for ax,(key,label) in zip(axes,metrics):
        vals=[result[n][key] for n in names]
        colors=['#e85d04' if n=='HGT-ALNS' else '#8da0ae' for n in names]
        bars=ax.bar(range(len(names)),vals,color=colors)
        ax.set_xticks(range(len(names)),names,rotation=32,ha='right');ax.set_ylabel(label);ax.set_title(label)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.0f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Unified static task-assignment comparison (3 paired scenarios)',fontweight='bold')
    fig.tight_layout();Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180,bbox_inches='tight');print(f'Saved: {a.output}')
if __name__=='__main__':main()
