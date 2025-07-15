import h5py
import numpy as np
import os.path as op
import pandas as pd
import paths
import weighting

nsamp = 1024

if __name__ == '__main__':
    rng = np.random.default_rng(72722818822976975902202257577628350481)

    m1, q, z, pdraw, ndraw = weighting.extract_selection_samples(op.join(paths.data, 'endo3_bbhpop-LIGO-T2100113-v12.hdf5'), nsamp, desired_pop_wt=weighting.default_pop_wt, rng=rng)

    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'pdraw': pdraw, 'ndraw': ndraw})
    df.to_hdf(op.join(paths.data, 'selection-samples.h5'), 'samples')

