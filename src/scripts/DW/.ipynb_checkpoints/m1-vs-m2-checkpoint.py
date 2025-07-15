import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import pandas as pd
import seaborn as sns

if __name__=='__main__':
    samples = pd.read_hdf(op.join(paths.data, 'pe-samples.h5'), key='samples')

    events = samples.groupby('evt')

    with sns.color_palette('husl', n_colors=len(events)):
        for evt, samps in events:
            m2 = samps['m1']*samps['q']
            sns.kdeplot(x=samps['m1'], y=m2, levels=[0.1, 0.5], alpha=0.25)
        
        plt.xlabel(r'$m_1 / M_\odot$')
        plt.ylabel(r'$m_2 / M_\odot$')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlim(5)
        plt.ylim(5)

        plt.tight_layout()
        plt.savefig(op.join(paths.figures, 'm1-vs-m2.pdf'))