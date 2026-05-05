import os
import scipy.io as sio
import h5py

case_dir = "/SSDHome/share/4dFlow/ChallengeData/TaskR1&R2/ValidationSet/Aorta/Center007/GE_30T_Architect/P076"

files = [
    "kdata_full.mat",
    "kdata_ktGaussian10.mat",
    "usmask_ktGaussian10.mat",
    "coilmap.mat",
    "segmask.mat",
]

def inspect_mat(path):
    print(f"\n=== {path} ===")
    try:
        data = sio.loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print("scipy keys:", keys)
        for k in keys:
            v = data[k]
            try:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            except Exception:
                print(f"  {k}: type={type(v)}")
    except Exception as e:
        print("scipy failed:", repr(e))
        try:
            with h5py.File(path, "r") as f:
                print("h5py keys:", list(f.keys()))
                for k in f.keys():
                    obj = f[k]
                    try:
                        print(f"  {k}: shape={obj.shape}, dtype={obj.dtype}")
                    except Exception:
                        print(f"  {k}: type={type(obj)}")
        except Exception as e2:
            print("h5py failed:", repr(e2))

for fn in files:
    inspect_mat(os.path.join(case_dir, fn))