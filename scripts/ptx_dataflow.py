#!/usr/bin/env python3
"""Symbolic dataflow over the f16/f32 arithmetic of one PTX entry: builds expression strings for registers
so a reduction tree or epilogue formula can be read exactly (op order and rounding points) without
hand-tracing thousands of lines. Integer/address ops are treated as opaque leaves."""
import re,sys,struct
def f32c(v): return struct.unpack('<f',struct.pack('<I',int(v)&0xffffffff))[0]
def load(path,entry):
    import sys,os; sys.path.insert(0,os.path.dirname(__file__))
    from ptx_int_emu import Kernel
    k=Kernel(open(path).read(),entry)
    return [s.strip("{} ").strip() for s in k.instrs]
class Flow:
    def __init__(self,lines,start=0,end=None):
        self.defs={}   # reg -> last (op, [args], lineno)
        self.alldefs={} # reg -> [(lineno, op, args)] in order
        self.order=[]
        for ln,s in enumerate(lines[start:end],start):
            if s.startswith(("@","$","//",".","bra","ret","bar","mbarrier","cp.async","st.","setp","ld.param","nanosleep","elect")): continue
            m=re.match(r"([a-z][a-zA-Z0-9_.:]*)\s+(.*)",s)
            if not m: continue
            op,rest=m.groups(); args=[a.strip() for a in re.split(r",(?![^{]*\})",rest)]
            if op.startswith("mma"): 
                d=args[0].strip("{}").split(","); 
                for k,r in enumerate(d): self._def(r.strip(),"mma"+str(k),[a for a in args[1:]],ln)
                continue
            if op.startswith("mov.b32") and args[1].startswith("{"):
                self._def(args[0],"pack",[a.strip() for a in args[1].strip("{}").split(",")],ln); continue
            if op.startswith("mov.b32") and args[0].startswith("{"):
                names=[a.strip() for a in args[0].strip("{}").split(",")]
                for k,r in enumerate(names): self._def(r,"unpack"+("lo" if k==0 else "hi"),[args[1]],ln)
                continue
            if op.startswith(("mov","cvt","add","sub","mul","fma","max","min","abs","neg","shfl","rsqrt","rcp","sqrt","ex2","prmt","shl","shr","and","or","xor","mad","ld","red","selp")):
                d=args[0].split("|")[0]
                self._def(d,op,args[1:],ln)
    def _def(self,r,op,args,ln):
        self.defs[r]=(op,args,ln); self.alldefs.setdefault(r,[]).append((ln,op,args)); self.order.append(r)
    def resolve(self,reg,at):
        """latest definition of reg strictly before statement index `at` (None = last)."""
        ds=self.alldefs.get(reg)
        if not ds: return None
        if at is None: return ds[-1]
        prev=[d for d in ds if d[0]<at]
        return prev[-1] if prev else None
    def expr(self,reg,depth=0,maxdepth=12,leaves=None,at=None):
        reg=reg.strip()
        if leaves and reg in leaves: return leaves[reg]
        d=self.resolve(reg,at)
        if d is None: return reg if reg.startswith("%") else self.const(reg)
        ln,op,args=d
        if depth>=maxdepth: return f"{reg}"
        if op.startswith("mma"): return f"MMA{op[3:]}@{ln}"
        if op.startswith("ld"): return f"LD@{ln}"
        if op.startswith("mov"): return self.expr(args[0],depth,maxdepth,leaves,at=ln)
        if op.startswith("cvt.rn.f16.f32"): return f"h({self.expr(args[0],depth+1,maxdepth,leaves,at=ln)})"
        if op.startswith("cvt.f32.f16"): return f"f({self.expr(args[0],depth+1,maxdepth,leaves,at=ln)})"
        sub=[self.expr(a,depth+1,maxdepth,leaves,at=ln) for a in args]
        short=op.replace(".f16x2","").replace(".rn","").replace(".ftz","").replace(".f32","32").replace(".sync.bfly.b32","_bfly").replace(".sync.idx.b32","_idx")
        return f"{short}({', '.join(sub)})"
    def const(self,tok):
        try:
            v=int(tok); 
            if abs(v)>1000: return f"{f32c(v):.6g}"
            return str(v)
        except: return tok
if __name__=="__main__":
    path,entry=sys.argv[1],sys.argv[2]; regs=sys.argv[3:]
    lines=load(path,entry); fl=Flow(lines)
    for r in regs: print(r,"=",fl.expr(r,maxdepth=int(__import__('os').environ.get("DEPTH","14"))))
