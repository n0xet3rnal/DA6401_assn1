import numpy as np
import os
_orig_np_load = np.load
def _custom_np_load(file, *args, **kwargs):
    print("Monkeypatch called!")
    return _orig_np_load(file, *args, **kwargs)
np.load = _custom_np_load

print("Loading...")
np.load('src/best_model.npy', allow_pickle=True)
print("Done!")
