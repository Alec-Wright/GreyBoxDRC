

def main(config_set=1):

    if config_set == 1:
        d = {
            'loss_func': 'rmsm',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'OnePoleAttRel', 'cond': False},
            'make_up': {'type': 'Static'}
        }

    elif config_set == 2:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'OnePoleAttRel', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 4}
        }
    else:
        print('no config')

    return d