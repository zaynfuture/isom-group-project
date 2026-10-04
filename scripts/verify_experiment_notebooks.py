"""Execute a full smoke notebook and retain outputs separately from source."""
import argparse
import json
import os
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('notebook',choices=['01_merchant_experiment','02_forecast_experiment'])
    args=parser.parse_args()
    os.environ['SPENDLENS_SMOKE']='1'
    os.environ.setdefault('SPENDLENS_DATASET','legacy')
    source=ROOT/'notebooks'/f'{args.notebook}.ipynb'
    destination=ROOT/'outputs/notebook-checks'
    destination.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.read(source,as_version=4)
    started=time.perf_counter()
    report={'notebook':source.name,'mode':'smoke','dataset':os.environ['SPENDLENS_DATASET'],'passed':False}
    try:
        client=NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        report['passed']=True
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        nbformat.write(notebook,destination/source.name)
        (destination/f'{args.notebook}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)

if __name__=='__main__': main()
