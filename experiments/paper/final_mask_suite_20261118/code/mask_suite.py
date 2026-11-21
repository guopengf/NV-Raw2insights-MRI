"""Four encoding-shared retrospective mask families with one point-budget rule."""
import argparse
import json
from pathlib import Path
import secrets

import numpy as np

import gaussian
import layered_mask
import pr4d_light
import pseudospiral


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('gaussian', 'pr4d', 'pseudo', 'uih')
PEPER = dict(model='proud_carotid', radius_power=2, turns=3,
             points_per_arm=100, rotation_deg=23.63)


def centered_acs(shape, size):
    if len(size) != 2 or any(not 1 <= s <= n for s, n in zip(size, shape)):
        raise ValueError('ACS dimensions must be positive and fit [Nz,Ny]')
    start = [n//2-s//2 for n,s in zip(shape,size)]
    out = np.zeros(shape, bool)
    out[start[0]:start[0]+size[0], start[1]:start[1]+size[1]] = True
    return out


def uih_pattern(shape, count, R, seed):
    """Shrink the observed layered footprint to fit the requested point budget."""
    nz, ny = shape
    profile = json.loads((ROOT/'uih_profile.json').read_text())
    native_z, native_y = profile['fit_shape']
    scale = min(1., np.sqrt(count/(nz*ny) / (911/(native_z*native_y))))
    base = profile['acs_bounds']
    ring_base = profile['ring_bounds']
    outside = np.ones(shape, bool)
    if ny == native_y:
        low, high = profile['support_bbox'][2:]
        outside[:, :low] = False
        outside[:, high:] = False
    for shrink in (1., .9, .8, .7, .6, .5):
        factor = scale * shrink
        sizes = [max(3, round((base[i+1]-base[i])*factor)) for i in (0,2)]
        starts = [n//2-s//2 for n,s in zip(shape,sizes)]
        ends = [a+s for a,s in zip(starts,sizes)]
        bounds = [starts[0],ends[0],starts[1],ends[1]]
        width = max(1, round(profile['acs_checker_width']*factor))
        rb = []
        for i,(n, old_n) in enumerate(zip(shape,profile['fit_shape'])):
            center = n//2
            lo = center + round((ring_base[2*i]-old_n//2)*n/old_n*factor)
            hi = center + round((ring_base[2*i+1]-old_n//2)*n/old_n*factor)
            margin = width if i == 1 else 0
            rb.extend((max(0,min(lo,starts[i]-margin)),min(n,max(hi,ends[i]+margin))))
        phase = tuple((profile['phase'][i]+starts[i]-base[2*i]) % len_axis
                      for i,len_axis in enumerate(np.shape(profile['grid_tile'])))
        options = dict(acs_bounds=bounds,ring_bounds=rb,grid_mode='periodic',
                       grid_tile=profile['grid_tile'],grid_phase=phase,
                       acs_checker_width=width,support=outside)
        acs,grid,_ = layered_mask.make_regions(ny,nz,**options)
        # Keep at least 15% of the budget for a seed-dependent outside pattern.
        if int(acs.sum()+grid.sum()) > count-max(1, round(.15*count)):
            continue
        mask,meta = layered_mask.generate_mask(ny,nz,1,R,seed,counts=[count],
                                               min_distance=profile['min_distance'],
                                               density_sigma=profile['density_sigma'],**options)
        meta.update(profile_scale=float(factor), design_support='UIH observed ky bounds at Ny=84; full grid otherwise')
        return mask[0,0],meta
    raise ValueError('Budget too small for a full ACS, both checkerboard bands and randomized outer UIH points')


def generate(method, ny, nz, nt, R, seed=None, *, acs_size=(3,3), counts=None,
             pseudo_profile='aorta_c001_R10'):
    """Return native binary [Z,Y] for UIH, otherwise [T,Z,Y]."""
    if method not in METHODS or not np.isfinite(R) or not 10 <= R <= 50:
        raise ValueError('Select a retained method and R in [10,50]')
    ny,nz,nt = (pseudospiral._integer(n,k) for n,k in zip((ny,nz,nt),('ny','nz','nt')))
    seed = secrets.randbits(63) if seed is None else pseudospiral._integer(seed,'seed',0)
    t_count = 1 if method=='uih' else nt
    if method!='uih' and nt<2:
        raise ValueError('Time-varying families require at least two cardiac phases')
    budget = np.full(t_count, int(np.floor(ny*nz/R)), dtype=int) if counts is None else np.asarray(counts)
    if (budget.shape!=(t_count,) or not np.issubdtype(budget.dtype,np.integer)
            or np.any((budget<1)|(budget>=ny*nz))):
        raise ValueError(f'counts must contain {t_count} positive integer point counts below Ny*Nz')
    if method == 'uih':
        pattern,meta = uih_pattern((nz,ny),int(budget[0]),R,seed)
    else:
        fixed = centered_acs((nz,ny),acs_size)
        if fixed.sum()>=budget.min():
            raise ValueError('ACS must leave at least one free point in every frame')
        if method=='gaussian':
            pattern,meta = gaussian.generate_pattern((nz,ny),budget,seed,fixed)
        elif method=='pr4d':
            mask,meta = pr4d_light.generate_mask(ny,nz,nt,R,seed,counts=budget,
                                                acs=fixed,radial_jitter=.0025,angle_jitter_deg=2)
            pattern = mask[0]
        else:
            configs = json.loads((ROOT/'pseudo_profiles.json').read_text())
            params = PEPER if pseudo_profile=='peper' else configs.get(pseudo_profile)
            if params is None:
                raise ValueError(f'Unknown pseudo profile: {pseudo_profile}')
            pattern,meta = pseudospiral.generate_pattern(ny,nz,nt,R,seed,counts=budget,
                                                         acs=fixed,radial_jitter=.0025,
                                                         arm_angle_jitter_deg=2,return_metadata=True,**params)
            meta['pseudo_profile'] = pseudo_profile
        if np.any(np.all(pattern[1:]==pattern[:-1], axis=(1,2))):
            raise RuntimeError('Adjacent frames have identical masks for this seed and budget')
        if not pattern[:,fixed].all():
            raise RuntimeError('Generated mask missed the explicit ACS')
        meta['acs_size_ZY'] = list(acs_size)
    planes = pattern.reshape(t_count,nz,ny)
    if not np.array_equal(planes.sum((1,2)),budget):
        raise RuntimeError('Generated counts differ from the requested budgets')
    meta.update(method=method,axes='ZY' if method=='uih' else 'TZY',shape=list(pattern.shape),
                seed=seed,requested_R_grid=float(R),effective_R_grid=(ny*nz/budget).tolist(),
                counts=budget.tolist(),nt_project=nt,encodings_shared=True,time_shared=method=='uih')
    return pattern,meta


def project_mask(pattern, nt):
    """Explicit [1,Nt,1,Z,Y,1] for the existing windowed data converter."""
    native = pattern if pattern.ndim == 3 else np.broadcast_to(pattern,(nt,*pattern.shape))
    if native.shape[0]!=nt:
        raise ValueError('Native mask frame count and target Nt differ')
    return native[None,:,None,:,:,None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    for name in ('ny','nz','nt','seed'):
        parser.add_argument('--'+name,type=int,required=name!='seed')
    parser.add_argument('--R',type=float,required=True)
    parser.add_argument('--acs',type=int,nargs=2,metavar=('Z','Y'))
    parser.add_argument('--counts',type=int,nargs='+')
    parser.add_argument('--pseudo-profile',default='aorta_c001_R10')
    parser.add_argument('--format',choices=('native','project'),default='native')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.method=='uih' and args.acs is not None:
        parser.error('UIH scales its own ACS; --acs applies to time-varying methods')
    mask,meta = generate(args.method,args.ny,args.nz,args.nt,args.R,args.seed,
                         acs_size=args.acs or (3,3),counts=args.counts,pseudo_profile=args.pseudo_profile)
    if args.format=='project':
        mask = project_mask(mask,args.nt)
        meta.update(serialized_axes='ETCZYX',serialized_shape=list(mask.shape))
    layered_mask.write_mask(args.output,mask,meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
