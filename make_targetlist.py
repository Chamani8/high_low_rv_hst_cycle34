import os
import numpy as np
from astropy.io import fits


if __name__ == "__main__":  
    path = os.path.expanduser("~")
    almaII_file = f"{path}alma2_catalogue.fit"

    fname = f"{path}alma3plusgaiaxp_extremeRV.fits"

    with fits.open(fname) as hdul:
        almaIII = hdul[1].data

    with fits.open(almaII_file) as hdul:
        almaII = hdul[1].data
        
    for star in almaIII:
        if "16" in star['ALS'] and '708' in star['ALS']:
            print(star['STARS'], star['ALS'])


    lines = ['#starname \t SpType\n']
    for star in almaIII:
        starname = star['ALS'].replace('LS ', 'LS').replace(' ', '-').replace('GL', 'AL')

        IImask = np.isin(almaII['GaiaDR2'], star['GAIA'])
        almaII_star = almaII[IImask]
        sptype = almaII_star['SpType'][0]#.split(', ')

        lines.append(f"{starname}\t{sptype}\n")

    with open(f"{path}target_list.dat", "w") as f:
        f.writelines(lines)