"""E4M3 / half helpers shared by the NR ViT reference tools (numpy)."""
import numpy as np
def e4m3_table():
    t=np.zeros(256,dtype=np.float32)
    for b in range(256):
        s=-1.0 if b&0x80 else 1.0; e=(b>>3)&15; m=b&7
        if e==0: v=(m/8)*2.0**-6
        elif e==15 and m==7: v=float('nan')
        else: v=(1+m/8)*2.0**(e-7)
        t[b]=s*v
    return t
E4M3=e4m3_table()
def decode(b): return E4M3[np.frombuffer(bytes(b),dtype=np.uint8)]
def weight_index(k,n):
    return (n//16)*512+(n%8)*64+((k%16)//4)*16+((n%16)//8)*8+k%4+(k//16)*4
def encode_e4m3(x):
    """RNE encode of float32 array to E4M3 bytes (satfinite: |x|>=464 -> 448, NaN -> NaN code)."""
    x=np.asarray(x,dtype=np.float32); out=np.zeros(x.shape,dtype=np.uint8)
    fin=E4M3[:127]  # positive finite codes 0..126
    a=np.abs(x); idx=np.searchsorted(fin,a)          # first code with value >= a
    idx=np.clip(idx,0,126); lo=np.clip(idx-1,0,126)
    dlo=a-fin[lo]; dhi=fin[idx]-a
    pick=np.where(dhi<dlo,idx,np.where(dlo<dhi,lo,np.where((lo&1)==0,lo,idx)))  # ties to even code
    pick=np.where(a>=fin[126],126,pick)
    out=pick.astype(np.uint8)|np.where(np.signbit(x),0x80,0).astype(np.uint8)
    out[np.isnan(x)]=0x7f
    return out
