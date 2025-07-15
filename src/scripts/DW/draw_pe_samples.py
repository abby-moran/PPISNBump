from glob import glob
import h5py
import numpy as np
import os.path as op
import pandas as pd
import paths
import re
import tqdm
import weighting

nsamp = 128

if __name__ == '__main__':
    rng = np.random.default_rng(232970088789901018827685773729153268726)
    df = None

    for f in tqdm.tqdm(glob(op.join(paths.data, 'pe-samples-raw', '*.h5'))):
        gwname = re.match('^.*(GW[0-9_]+[0-9]+).*.h5$', f)[1]

        try:
            m1, q, z, wt = weighting.extract_posterior_samples(f, nsamp, desired_pop_wt=weighting.default_pop_wt, rng=rng)
        except ValueError as err:
            print(f'could not process {gwname}: exception {err}')
        d = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'wt': wt, 'evt': gwname})
        if df is None:
            df = d
        else:
            df = pd.concat((df, d), ignore_index=True)

    df.to_hdf(op.join(paths.data, 'pe-samples.h5'), key='samples', mode='w')
