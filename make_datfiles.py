from astroquery.simbad import Simbad
from astropy.io import fits

if __name__ == "__main__":
    path = "/Users/cgunasekera/forked_extstar_data/high_low_rv_hst_cycle34/"
    dat_path = f"{os.path.expanduser("~")}/extstar_data/DAT_files/"
    almaIII_file = f"{path}alma3plusgaiaxp.fits"
    crossmatch_file = almaIII_file.replace('.fits', '_wsptype.fits')
    RV_select_file = crossmatch_file.replace("_wsptype.fits", "_extremeRV.fits")

    with fits.open(RV_select_file) as hdul:
        almaIII_hiRV = hdul[1].data

    s = Simbad()
    s.add_votable_fields('U', 'B', 'V', 'J', 'H', 'K', 'sp_type', 'sp_bibcode')

    for i, star in enumerate(almaIII_hiRV):
        if star['ALS'] in ['GLS 606', 'GLS 6268', 'ALS 19 709', 'GLS 8415']:
            continue
        if star['STARS'] in ['HD 36 982']:
            continue

        starname = star['ALS'].lower().replace('ls ', 'ls').replace(' ', '-').replace('gl', 'al')
        dat_filename = f"{starname}.dat"
            
        lines = [f"# data file for observations of {star['ALS'].replace('GL', 'AL')}\n"]

        result = s.query_object(star['SIMBAD'])

        photo_bands = ['U', 'B', 'V', 'J', 'H', 'K']
        #print(i, star['ALS'], dat_filename)
        for band in photo_bands:
            if len(result[band].value) > 0:
                if np.ma.is_masked(result[band][0]): 
                    print(star['STARS'], dat_filename, band)
                    continue
                #print(i, star['SIMBAD'], result['U'][0]) #len(result['U'].value), result['U'].value, result['B'].value
                lines.append(f"{band} = {result[band][0]:.3f} +/- 0.05;\n")
            else:
                print(star['STARS'], dat_filename, band)
            
        if len(result['sp_type']) > 0:
            lines.append(f'sptype = {result['sp_type'][0]}; ref = {result['sp_bibcode'][0]}')

        with open(f"{dat_path}{dat_filename}", "w") as f:
            f.writelines(lines)
