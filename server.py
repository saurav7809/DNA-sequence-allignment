"""
Flask API backend for DNA Alignment Web Dashboard
Run: python server.py
Then open: http://localhost:5000
"""

import os, sys, io, time, random, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

MATCH    =  2
MISMATCH = -1
GAP      = -2

def random_dna(n, seed=42):
    rng = random.Random(seed)
    return ''.join(rng.choice("ACGT") for _ in range(n))

def nw_with_matrix(seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=np.int32)
    dp[0,:] = np.arange(n+1)*gap
    dp[:,0] = np.arange(m+1)*gap
    for i in range(1,m+1):
        for j in range(1,n+1):
            d=dp[i-1,j-1]+(match if seq1[i-1]==seq2[j-1] else mismatch)
            u=dp[i-1,j]+gap; l=dp[i,j-1]+gap
            dp[i,j]=max(d,u,l)
    return int(dp[m,n]), dp.tolist()

def nw_wavefront(seq1, seq2, match=MATCH, mismatch=MISMATCH, gap=GAP):
    m, n = len(seq1), len(seq2)
    arr1=np.frombuffer(seq1.encode(),dtype=np.uint8)
    arr2=np.frombuffer(seq2.encode(),dtype=np.uint8)
    dp=np.zeros((m+1,n+1),dtype=np.int32)
    dp[0,:]=np.arange(n+1)*gap; dp[:,0]=np.arange(m+1)*gap
    diagonals=[]
    for diag in range(2,m+n+1):
        rows=np.arange(max(1,diag-n),min(m,diag-1)+1,dtype=np.intp)
        cols=diag-rows
        mm=(arr1[rows-1]==arr2[cols-1])
        sd=dp[rows-1,cols-1]+np.where(mm,match,mismatch)
        su=dp[rows-1,cols]+gap; sl=dp[rows,cols-1]+gap
        dp[rows,cols]=np.maximum(sd,np.maximum(su,sl))
        diagonals.append([[int(r),int(c)] for r,c in zip(rows,cols)])
    return int(dp[m,n]), diagonals, dp.tolist()

def measure(fn,*args):
    t0=time.perf_counter(); r=fn(*args)
    return r,(time.perf_counter()-t0)*1000

@app.route('/')
def index():
    return send_from_directory('web','index.html')

@app.route('/api/align',methods=['POST'])
def align():
    data=request.json
    seq1=data.get('seq1','ACGT').upper().strip()
    seq2=data.get('seq2','ACT').upper().strip()
    match=int(data.get('match',MATCH))
    mismatch=int(data.get('mismatch',MISMATCH))
    gap=int(data.get('gap',GAP))
    if len(seq1)>30 or len(seq2)>30:
        return jsonify({'error':'Max 30 chars for matrix view. Use benchmark for large sequences.'}),400
    (cs,cm),cpu_ms=measure(nw_with_matrix,seq1,seq2,match,mismatch,gap)
    (ws,diags,wm),wf_ms=measure(nw_wavefront,seq1,seq2,match,mismatch,gap)
    return jsonify({'seq1':seq1,'seq2':seq2,'cpu_score':cs,'wf_score':ws,
        'cpu_ms':round(cpu_ms,4),'wf_ms':round(wf_ms,4),
        'match':match,'mismatch':mismatch,'gap':gap,
        'matrix':cm,'diagonals':diags,
        'speedup':round(cpu_ms/wf_ms,3) if wf_ms>0 else 0,
        'validation':cs==ws})

@app.route('/api/benchmark',methods=['POST'])
def benchmark():
    data=request.json or {}
    sizes=data.get('sizes',[100,250,500,750,1000])
    trials=data.get('trials',2)
    match=int(data.get('match',MATCH))
    mismatch=int(data.get('mismatch',MISMATCH))
    gap=int(data.get('gap',GAP))
    results=[]
    for sz in sizes:
        ct,wt=[],[]
        for t in range(trials):
            s1=random_dna(sz,100+t); s2=random_dna(sz,200+t)
            (cs,_),cm=measure(nw_with_matrix,s1,s2,match,mismatch,gap)
            (ws,_,_),wm=measure(nw_wavefront,s1,s2,match,mismatch,gap)
            ct.append(cm); wt.append(wm)
        ac=sum(ct)/trials; aw=sum(wt)/trials
        results.append({'size':sz,'cpu_ms':round(ac,3),'wf_ms':round(aw,3),
            'speedup':round(ac/aw,3) if aw>0 else 0,'validation':cs==ws})
    return jsonify({'results':results})

@app.route('/api/random',methods=['POST'])
def gen_random():
    data=request.json or {}
    length=min(int(data.get('length',10)),30)
    return jsonify({'seq1':random_dna(length,42),'seq2':random_dna(length,99)})

if __name__=='__main__':
    print("\n"+"="*50)
    print("  DNA Alignment Web Dashboard")
    print("  Open: http://localhost:5000")
    print("="*50+"\n")
    app.run(debug=False,port=5000)
