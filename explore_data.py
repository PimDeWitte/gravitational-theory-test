import h5py
import numpy as np

file_path = 'data/15yr_cw_analysis/data/15yr_quickCW_detection.h5'
with h5py.File(file_path, 'r') as f:
    print('Keys in HDF5 file:', list(f.keys()))
    
    # Check parameter names
    if 'par_names' in f:
        par_names = f['par_names'][()]
        print('\nSearching for J2043+1711 parameters:')
        j2043_indices = []
        for i, name in enumerate(par_names):
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if 'J2043' in name or '2043' in name:
                print(f'Index {i}: {name}')
                j2043_indices.append(i)
        
        if not j2043_indices:
            print('J2043+1711 not found in this file. Listing all pulsar names:')
            pulsar_names = set()
            for name in par_names:
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                # Extract pulsar name from parameter
                if '_' in name and not name.startswith('0_'):
                    pulsar = name.split('_')[0]
                    pulsar_names.add(pulsar)
            print('Pulsars in dataset:', sorted(pulsar_names))
    
    # Check samples for J2043 if found
    if 'samples_cold' in f and j2043_indices:
        samples = f['samples_cold'][()]
        print(f'\nSamples shape: {samples.shape}')
        print(f'J2043+1711 parameter samples (first 5):')
        for idx in j2043_indices:
            param_name = par_names[idx].decode('utf-8') if isinstance(par_names[idx], bytes) else par_names[idx]
            print(f'{param_name}: {samples[:5, idx]}') 