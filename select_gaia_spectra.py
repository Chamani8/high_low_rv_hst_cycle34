import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astroquery.simbad import Simbad
from astropy.table import Table
import re
from scipy.interpolate import interp1d
import astropy.units as u
from synphot.models import Empirical1D
from synphot import SpectralElement, SourceSpectrum, Observation

def sp_in_range(sp, start, end): 
    if sp == '':
        return False
    
    in_range = True
    if sp[0] not in [start[0], end[0]]:
        return False

    try:
        num = int(sp[1])
    except:
        return False

    if sp[0] == start[0] and int(sp[1]) < int(start[1]):
        in_range = False

    if sp[0] == end[0] and int(sp[1]) > int(end[1]):
        in_range = False

    return in_range


def select_spType(star_tab_wSpType, start, end):
    # Select all the Alma II stars with a spectra type starting with O or B:
    sp = star_tab_wSpType['SpType'].astype(str)  # convert to string
    first_letter = np.char.upper(np.char.strip(sp)).astype(str)
    first_letter = np.array([s[0] if len(s) > 0 else '' for s in first_letter])

    # Make the mask
    sp_mask_OB = np.isin(first_letter, ['O', 'B'])
    star_tab_wSpType_OB = star_tab_wSpType[sp_mask_OB]

    filtered_rows = []
    for star in star_tab_wSpType_OB:
        sptype = star['SpType'].strip()
        sptype = sptype.split(', ')
        if len(sptype) > 1:
            for sp in sptype:
                #print(sptype, sp)
                if sp_in_range(sp, start, end):
                    #print(sp, sp_in_range(sp, start, end)) 
                    filtered_rows.append(star)
                    break
                else:
                    #print(sp, sp_in_range(sp, start, end)) 
                    continue
        else:
            if sp_in_range(sptype[0], start, end):
                filtered_rows.append(star)
            #else:
            #    print(sptype)
    
    star_OB_selected = Table(rows=filtered_rows, names=star_tab_wSpType.columns.names)
    return star_OB_selected


def synthetic_obs_BV(wavelength, flux, B_filter, V_filter):
    # Clip negative flux
    flux = np.clip(flux, 1e-20, None)

    # Create SourceSpectrum
    spectrum = SourceSpectrum(
        Empirical1D,
        points=wavelength * u.AA,
        lookup_table=flux * u.erg / (u.s * u.cm**2 * u.AA)
    )

    #integrate spectrum over filter
    obs_B = Observation(spectrum, B_filter, force='taper') #F_B
    obs_V = Observation(spectrum, V_filter, force='taper') #F_V

    # Compute AB magnitudes
    try:
        m_B = obs_B.effstim('abmag')
        m_V = obs_V.effstim('abmag')
    except Exception:
        return np.nan, np.nan, np.nan
    
    m_B = m_B.value - (-0.09)   # m_B + 0.09 vega offset
    m_V = m_V.value - 0.02      # m_V - 0.02 vega offset

    return m_B, m_V, m_B - m_V


def get_intrinsic_colors(sptype):
    # Early-type OB stars, approximate colors
    base_points_BV = {9.0:-0.31, 10.0:-0.30, 11.0:-0.27, 12.0:-0.24, 13.0:-0.20}

    # Standard intrinsic V-J, V-H, V-Ks for OB stars
    base_VJ = {9.0:-0.87, 10.0:-0.83, 11.0:-0.74, 12.0:-0.66, 13.0:-0.56}
    base_VH = {9.0:-0.97, 10.0:-0.92, 11.0:-0.85, 12.0:-0.79, 13.0:-0.72}
    base_VKs = {9.0:-1.00, 10.0:-0.97, 11.0:-0.93, 12.0:-0.89, 13.0:-0.85}

    lum_class_correction = {"V":0.00,"IV":0.01,"III":0.02,"II":0.03,"I":0.05}
    
    # Regex to parse spectral type
    pattern = re.compile(r'([OB])\s*([0-9./-]+)\s*(?:([IV]+[ab]*(?:/[IV]+[ab]*)?))?')
    
    if not isinstance(sptype, str) or "e" in sptype.lower() or "p" in sptype.lower():
        return None, None, None, None
    
    match = pattern.match(sptype.strip().upper())
    if not match:
        return None, None, None, None
    
    sp_class, subtype_str, lum_str = match.groups()

    if lum_str is None or not lum_str.startswith("V"):
        return None, None, None, None

    try:
        if "/" in subtype_str:
            parts = [float(p) for p in subtype_str.split("/")]
            subtype = np.mean(parts)
        elif "-" in subtype_str:
            parts = [float(p) for p in subtype_str.split("-")]
            subtype = np.mean(parts)
        else:
            subtype = float(subtype_str)
    except ValueError:
        return None, None, None, None

    num = subtype if sp_class == "O" else 10 + subtype
    
    if num < 9.0 or num > 13.0:
        return None, None, None, None
    
    lum_correction = 0.0
    if lum_str:
        lum_parts = lum_str.split("/")
        corrections = []
        for l in lum_parts:
            l = l.strip()
            if l.startswith("I"): l = "I"
            elif l.startswith("II"): l = "II"
            elif l.startswith("III"): l = "III"
            elif l.startswith("IV"): l = "IV"
            elif l.startswith("V"): l = "V"
            corrections.append(lum_class_correction.get(l, 0.0))
        lum_correction = np.mean(corrections)
    
    # Interpolate intrinsic colors
    x_base = np.array(sorted(base_points_BV.keys()))
    BV0 = np.interp(num, x_base, [base_points_BV[k] for k in x_base]) + lum_correction
    VJ0 = np.interp(num, x_base, [base_VJ[k] for k in x_base])
    VH0 = np.interp(num, x_base, [base_VH[k] for k in x_base])
    VK0 = np.interp(num, x_base, [base_VKs[k] for k in x_base])

    # Convert to colors relative to V (intrinsic)
    J_V0 = -VJ0
    H_V0 = -VH0
    Ks_V0 = -VK0

    return BV0, J_V0, H_V0, Ks_V0


def calc_AV_JHKs(V, J, H, Ks, JV0, HV0, KsV0, AJ_AV=0.269, AH_AV=0.163, AKs_AV=0.112, debug=False):
    """
    Compute A_V from J, H, Ks magnitudes using all three NIR colors with a robust linear fit.
    
    Parameters
    ----------
    J, H, Ks : float
        Observed NIR magnitudes
    JV0, HV0, KsV0 : float
        Intrinsic colors (J-V)_0, (H-V)_0 and (Ks-V)_0 for the spectral type
    AJ_AV, AH_AV, AKs_AV : float
        NIR-to-V extinction ratios (default values from Cardelli+1989)
    debug : bool
        If True, prints intermediate values

    Returns
    -------
    AV : float
        Extinction in V band
    """

    E_JV = (J - V) - JV0
    E_HV = (H - V) - HV0
    E_KV = (Ks - V) - KsV0

    if debug:
        print("E(J-V):", E_JV, "E(H-V):", E_HV, "E(Ks-V):", E_KV)
    
    AVs=[]
    for e_lv, alav in zip([E_JV, E_HV, E_KV], [AJ_AV, AH_AV, AKs_AV]):
        av = e_lv/(alav-1)
        AVs.append(av)
    
    AV = np.mean(AVs)
    
    return AV


if __name__ == "__main__":
    path = "/Users/cgunasekera/forked_extstar_data/high_low_rv_hst_cycle34/"
    almaIII_file = f"{path}alma3plusgaiaxp.fits"
    almaII_file = f"{path}alma2_catalogue.fit"
    crossmatch_file = almaIII_file.replace('.fits', '_wsptype.fits')

    do = "get BV" # options: "create_crossmatch_fits", "check_crossmatch_fits", "get BV"

    if do == "create_crossmatch_fits":
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('sp_type', 'sp_bibcode')

        with fits.open(almaIII_file) as hdul:
            almaIII = hdul[1].data

        with fits.open(almaII_file) as hdul:
            almaII = hdul[1].data

        #print(almaII.columns.names)
        #print(almaIII.columns.names)

        # Select all the Alma II stars with a spectra type:
        sp_mask = ~np.isin(almaII['SpType'], ['', 'B', 'O', 'OB', 'OB+', 'OB-', 'OB+M0'])
        almaII_wSpType = almaII[sp_mask]

        almaII_wSpType_OB = select_spType(almaII, start="O9", end="B3")

        #print(almaIII['GAIA'], almaII_wSpType['GaiaDR2'])

        # Select all the Alma III stars that are in Alma II and has spectral type:
        IImask = np.isin(almaIII['GAIA'], almaII_wSpType['GaiaDR2'])
        almaIII = almaIII[IImask]

        print(f"Selecting {len(almaIII['GAIA'])} of {len(almaII['GaiaDR2'])} stars.")

        sub_table = Table(rows=almaIII, names=almaIII.columns.names)
        sub_table.write(crossmatch_file, overwrite=True)
        print(f"new file created: {crossmatch_file}")

    elif do == "check_crossmatch_fits":
        with fits.open(crossmatch_file) as hdul:
            almaIII = hdul[1].data

        print(almaIII.columns.names)
        print(len(almaIII['GAIA']))
        print(almaIII['SIMBAD'])

    elif do == "get BV":
        band_path = "/Users/cgunasekera/measure_extinction/measure_extinction/data/Band_RespCurves/"

#        with fits.open(almaIII_file) as hdul:
#            almaIII = hdul[1].data

        # Load Gaia XP spectra
        with fits.open(crossmatch_file) as hdul:
            almaIII = hdul[1].data

        with fits.open(almaII_file) as hdul:
            almaII = hdul[1].data

        # Load B and V filter curves
        B_filter = SpectralElement.from_file(f"{band_path}JohnB.dat", wave_unit='angstrom')
        V_filter = SpectralElement.from_file(f"{band_path}JohnV.dat", wave_unit='angstrom')

        # Loop over stars
        high_Rv_counter = 0
        low_Rv_counter = 0
        high_RV_stars = []
        low_RV_stars = []
        for star in almaIII:
            wavelength = star['WAVE'] * 10  # Gaia XP: nm -> Å
            flux = star['FLUX']             # use raw flux, arbitrary units
            flux_err = star['FLUX_ERR']

            # Compute B, V synthetic magnitudes
            #try:
            B_mag, V_mag, BminusV = synthetic_obs_BV(wavelength, flux, B_filter, V_filter)
            #except Exception as e:
                #print(f"Skipping {star['STARS']}: {e}")
            #    continue

            # Print star, synthetic B, V, B-V, and Gaia BP-RP color for sanity check
            #print(star['STARS'], B_mag, V_mag, BminusV, 0.8 - 1.1 * (star['BP'] - star['RP']))

            IImask = np.isin(almaII['GaiaDR2'], star['GAIA'])
            almaII_star = almaII[IImask]
            sptype = almaII_star['SpType'][0].split(', ')

            RVs = []
            for sp in sptype:
                #print(sp)
                BV0, JV0, HV0, KV0 = get_intrinsic_colors(sp)
                if BV0 is None:
                    continue

                E_BV = BminusV - BV0
                E_VK = (V_mag - star['Ks']) - (-KV0)
                AV = calc_AV_JHKs(V_mag, star['J'], star['H'], star['KS'], JV0, HV0, KV0)
                RV = AV / E_BV
                RVs.append(RV)
                #print(star['STARS'], sp)
                #print(star['J'], star['H'], star['KS'], JV0, HV0, KV0)
                #print(E_BV, AV, RV)

            #print(star['STARS'], sptype, RVs, 1.1 * E_VK / E_BV)

            avg_RV = np.mean(RVs)
#            if star['STARS'] in ['GLS 15 273', 'GLS 18 106', 'GLS 6213', 'HDE 306 234']:
#                print(star['STARS'], sptype, np.mean(RVs), 1.1 * E_VK / E_BV, AV, E_BV, avg_RV > 5.0 and E_BV > 0.3)
#
#            continue

            if (avg_RV > 5.0 or avg_RV < 2.5) and E_BV > 0.3 and avg_RV > 1.0: #and avg_RV < 12.0:
                high_Rv_counter += 1
                #print(star['STARS'])
                high_RV_stars.append(star)

            #if avg_RV < 2.5 and avg_RV > 1.0 and E_BV > 0.3:
            #    low_RV_of_star = avg_RV
            #    low_Rv_counter += 1
            #    #print(star['STARS'])
            #    low_RV_stars.append(star['STARS'])
        
        
        print(f"RESULTS:\nfound {low_Rv_counter} RV < 2.5 stars, {high_Rv_counter} RV > 5.0 stars.")
        #print(high_RV_stars, low_RV_stars)
        #print(low_RV_of_star)

        #outname = crossmatch_file.replace("_wsptype.fits", "_lowRV.fits")
        #low_RV_table = Table(rows=low_RV_stars, names=almaIII.columns.names)
        #low_RV_table.write(crossmatch_file, overwrite=True)
        #print(f"new file created: {crossmatch_file}")

        outname = crossmatch_file.replace("_wsptype.fits", "_extremeRV.fits")
        high_RV_table = Table(rows=high_RV_stars, names=almaIII.columns.names)
        high_RV_table.write(outname, overwrite=True)
        print(f"new file created: {outname}")
