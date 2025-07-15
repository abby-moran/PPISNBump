from astropy.cosmology import Planck18
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import paths
import seaborn as sns
import weighting

if __name__ == '__main__':
    sns.set_palette('colorblind')

    rng = np.random.default_rng(278954249059388231756799694135579037370)

    obs = pd.read_hdf(op.join(paths.data, 'mock_observations.h5'), key='observations')
    obs['m1_det'] = obs['m1']*(1 + obs['z'])
    obs['dl'] = Planck18.luminosity_distance(obs['z'].to_numpy()).to(u.Gpc).value

    i = rng.integers(len(obs))
    row = obs.iloc[i]

    m1d, q, dl, wt = weighting.draw_mock_samples(row['log_mc_obs'], row['sigma_log_mc'], row['q_obs'], row['sigma_q'], row['log_dl_obs'], row['sigma_log_dl'], size=1000)

    pg = sns.PairGrid(pd.DataFrame({r'$m_{1,\mathrm{det}}$':m1d, r'$q$':q, r'$d_L / \mathrm{Gpc}$':dl}), diag_sharey=False)
    pg.map_diag(sns.kdeplot)
    pg.map_lower(sns.kdeplot)
    pg.map_upper(sns.scatterplot)

    for j, row_true in enumerate([row['m1_det'], row['q'], row['dl']]):
        for i, col_true in enumerate([row['m1_det'], row['q'], row['dl']]):
            if i == j:
                pg.axes[i,j].axvline(row_true, color='k')
            else:
                pg.axes[i,j].axvline(row_true, color='k')
                pg.axes[i,j].axhline(col_true, color='k')

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'mock_observation_corner.pdf'))