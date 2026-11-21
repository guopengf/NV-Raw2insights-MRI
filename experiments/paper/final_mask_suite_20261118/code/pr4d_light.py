"""OSU PR4D encoding-0 coordinates with optional gentle retrospective jitter.

Port of OSU-CMR/cmr-sampling pr4d_fun.m, commit63d3635531fe98e5c95ab11b54eb593448474451.
Copyright(c)2022 Mihir Joshi; MIT license and pinned source in ../upstream_osu/.
PR4D generates per-readout coordinates, not spiral arms. Only e0 is constructed;
the resulting TZY pattern is broadcast to four velocity encodings.
"""
import numpy as np


def matlab_round(x):return np.copysign(np.floor(np.abs(x)+.5),x).astype(np.int64)


def native_mask(ny,nz,nt,R,seed=0,*,radial_jitter=0,angle_jitter_deg=0):
    if (any(int(n)!=n or n<4 for n in [ny,nz]) or int(nt)!=nt or nt<1 or int(seed)!=seed or seed<0
            or not np.isfinite(R) or R<=1 or radial_jitter<0
            or not np.isfinite([radial_jitter,angle_jitter_deg]).all()
            or not 0<=angle_jitter_deg<=180):raise ValueError('Invalid PR4D dimensions/R/jitter')
    ny,nz,nt=int(ny),int(nz),int(nt)
    events=max(1,int(np.floor(ny*nz/R+.5)));M=max(2,nt*events)
    gap=.25*np.sqrt(2)+np.finfo(float).eps
    gr=2-(np.sqrt(5)+1)/2;ri=np.floor((max(ny,nz)-1)/2)
    j=np.arange(1,M+1)
    r=np.remainder(ri*(gr*(j-1)),ri)
    theta=2*np.pi*np.remainder(35**(1/3),1)*j
    angle=np.arctan2(np.sin(theta),np.cos(theta))
    x=r**(2.5/2)*np.cos(angle);z=r**(2.5/2)*np.sin(angle)
    cy=(ny/2-(gap+1))*x/max(abs(x))+gap*np.cos(angle)
    cz=(nz/2-(gap+1))*z/max(abs(z))+gap*np.sin(angle)
    if radial_jitter or angle_jitter_deg:
        rng=np.random.default_rng(seed)
        rho=np.hypot(cy/(ny/2),cz/(nz/2));phi=np.arctan2(cz/(nz/2),cy/(ny/2))
        if radial_jitter:rho=np.maximum(0,rho+rng.uniform(-radial_jitter,radial_jitter,M))
        if angle_jitter_deg:phi=phi+np.deg2rad(rng.uniform(-angle_jitter_deg,angle_jitter_deg,M))
        cy=ny/2*rho*np.cos(phi);cz=nz/2*rho*np.sin(phi)
    # Quantize centered coordinates exactly as the original MATLAB port.
    yy=(matlab_round(cy)+ny//2)[:nt*events];zz=(matlab_round(cz)+nz//2)[:nt*events]
    frame=np.repeat(np.arange(nt),events)
    valid=(yy>=0)&(yy<ny)&(zz>=0)&(zz<nz)
    if not (radial_jitter or angle_jitter_deg) and not valid.all():
        raise RuntimeError('Native PR4D produced out-of-grid coordinates')
    m=np.zeros((nt,nz,ny),dtype=np.uint8);m[frame[valid],zz[valid],yy[valid]]=1
    return m,int((~valid).sum())


def generate_mask(ny,nz,nt,R,seed=0,*,counts=None,radial_jitter=0,angle_jitter_deg=0,acs=None):
    native,rejected=native_mask(ny,nz,nt,R,seed,radial_jitter=radial_jitter,angle_jitter_deg=angle_jitter_deg)
    targets=np.full(nt,int(np.floor(ny*nz/R))) if counts is None else np.asarray(counts)
    if targets.shape!=(nt,) or not np.isfinite(targets).all() or np.any(targets!=np.floor(targets)) or np.any((targets<1)|(targets>ny*nz)):
        raise ValueError('counts must be integer [T], in[1,Ny*Nz]')
    fixed=np.zeros((nz,ny),bool) if acs is None else np.asarray(acs,dtype=bool)
    if fixed.shape!=(nz,ny) or fixed.sum()>=targets.min():
        raise ValueError('ACS must leave a free sample in every frame')
    out=native.astype(bool)|fixed;rounds=[]
    for k in range(1,9):
        missing=out.sum((1,2))<targets
        if not missing.any():break
        rate=max(2.,R/2**k)
        extra,nreject=native_mask(ny,nz,nt,rate,seed+100003*k,radial_jitter=radial_jitter,angle_jitter_deg=angle_jitter_deg)
        additions=(extra.astype(bool)&~out).sum((1,2))*missing
        out[missing]|=extra[missing].astype(bool)
        rounds.append(dict(R=rate,seed=seed+100003*k,added=additions.tolist(),out_of_grid=nreject))
    if np.any(out.sum((1,2))<targets):raise RuntimeError('PR4D family cannot meet requested counts')
    rng=np.random.default_rng(seed+700001)
    for t in range(nt):
        idx=np.flatnonzero(out[t]&~fixed);drop=int(out[t].sum())-int(targets[t])
        if drop:out[t].flat[rng.choice(idx,drop,replace=False)]=False
    pattern=out.astype(np.uint8)
    if nt>1 and np.all(targets<ny*nz) and np.array_equal(pattern,np.broadcast_to(pattern[:1],pattern.shape)):
        raise RuntimeError('PR4D output is static over time')
    mask=np.broadcast_to(pattern[None],(4,*pattern.shape))
    meta=dict(method='PR4D e0 shared, conditioned exact counts',
              source='https://github.com/OSU-CMR/cmr-sampling',commit='63d3635531fe98e5c95ab11b54eb593448474451',
              axes='ETZY',shape=list(mask.shape),seed=seed,radial_jitter=radial_jitter,angle_jitter_deg=angle_jitter_deg,
              jitter='per-readout continuous e0 polar-coordinate perturbation BEFORE rounding; not spiral-arm jitter; not vendor sequence',
              encoding_policy='generate e0 TZY once and broadcast four encodings; no encoding-specific jitter',
              counts=targets.tolist(),R_rect=(ny*nz/targets).tolist(),native_counts=native.sum((1,2)).tolist(),
              out_of_grid_discarded=rejected,supplementation=rounds,
              added_to_native=(pattern.astype(bool)&~native.astype(bool)).sum((1,2)).tolist(),
              removed_from_native=(native.astype(bool)&~pattern.astype(bool)).sum((1,2)).tolist())
    return mask,meta
