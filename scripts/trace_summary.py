import sys,struct,os,collections; sys.path.insert(0,os.path.dirname(__file__))
from ptx_addr_trace import trace_grid
CNT=bytearray([0x7f])*65536
def which(bases,a):
    for k,v in sorted(bases.items(),key=lambda x:-x[1]):
        if a>=v and a<v+(1<<26): return k,a-v
    return "?",a
def summarize(name,events,bases,lanes=2):
    by=collections.defaultdict(list)
    for e in events:
        kind=e[0]
        if kind=="cp.async": b,off=which(bases,e[3]); by[(kind,b)].append((e[2],off,e[5],e[4]))
        elif kind.startswith(("ld.global","st.global","red.global")): b,off=which(bases,e[3]); by[(kind,b)].append((e[2],off,e[4],None))
    print(f"== {name}: {len(events)} events")
    for k,v in sorted(by.items()):
        offs=[x[1] for x in v]; print(f"  {k[0]:10s} {k[1]:8s} n={len(v):5d} min={min(offs)} max={max(offs)} sizes={sorted(set(x[2] for x in v))} distinct_offs={len(set(offs))}")
        for w in range(4):
            w0=[x for x in v if x[0][1]==w and x[0][2]==0 and x[0][0]<lanes][:8]
            if w0: print(f"     warp{w} lanes<{lanes} first:",[(x[0][0],x[1])+((x[3],) if x[3] is not None else ()) for x in w0])
    return by
