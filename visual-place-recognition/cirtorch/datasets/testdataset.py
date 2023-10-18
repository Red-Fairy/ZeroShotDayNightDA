import os
import pickle

DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', '247tokyo1k', 'gp_dl_nr', 'gp_dr_nr']

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname

    if 'gp_' not in dataset:
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
    else:
        cfg['ext'] = ''
        cfg['qext'] = ''
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = 'data/test/GardensPointWalking/'

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])
