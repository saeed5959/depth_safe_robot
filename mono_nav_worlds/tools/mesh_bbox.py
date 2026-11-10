import sys,glob,numpy as np
from lxml import etree
def bbox(f):
    t=etree.parse(f); r=t.getroot(); ns={'c':r.nsmap[None]}
    unit=t.find('.//c:asset/c:unit',ns); s=float(unit.get('meter')) if unit is not None else 1
    up=(t.find('.//c:asset/c:up_axis',ns).text if t.find('.//c:asset/c:up_axis',ns) is not None else 'Y_UP')
    geoms={}
    for g in t.iterfind('.//c:library_geometries/c:geometry',ns):
        ps=[np.array(src.find('c:float_array',ns).text.split(),float).reshape(-1,3) for src in g.iterfind('.//c:source',ns) if 'pos' in src.get('id','').lower()]
        if ps: geoms[g.get('id')]=np.vstack(ps)
    out=[]
    def walk(node,M):
        for ch in node:
            tag=etree.QName(ch).localname
            if tag=='matrix': M=M@np.array(ch.text.split(),float).reshape(4,4)
            elif tag=='translate': T=np.eye(4);T[:3,3]=np.array(ch.text.split(),float);M=M@T
            elif tag=='scale': S=np.diag(list(map(float,ch.text.split()))+[1]);M=M@S
            elif tag=='rotate':
                x,y,z,a=map(float,ch.text.split());a=np.radians(a);k=np.array([x,y,z]);k/=np.linalg.norm(k)
                K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]]);R=np.eye(4);R[:3,:3]=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K;M=M@R
        for ch in node:
            tag=etree.QName(ch).localname
            if tag=='instance_geometry':
                p=geoms[ch.get('url')[1:]];h=np.c_[p,np.ones(len(p))]@M.T;out.append(h[:,:3])
            elif tag=='node': walk(ch,M)
    for n in t.iterfind('.//c:visual_scene/c:node',ns): walk(n,np.eye(4))
    p=np.vstack(out)*s
    if up=='Y_UP': p=np.c_[p[:,0],-p[:,2],p[:,1]]
    return p.min(0).round(2),p.max(0).round(2)
for m in sys.argv[1:]:
    f=glob.glob(f'/tmp/house/models/aws_robomaker_residential_{m}/meshes/*collision*')[0]
    lo,hi=bbox(f); print(f'{m:16s} size={(hi-lo).round(2)} min={lo} max={hi}')
