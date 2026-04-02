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
    
    if not isinstance(sptype, str):# or "e" in sptype.lower():# or "p" in sptype.lower():
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


def calc_AV_JHKs(V, J, H, Ks, JV0, HV0, KsV0, flux, AJ_AV=0.269, AH_AV=0.163, AKs_AV=0.112, debug=False):
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

    AV = []
    if abs(flux[-1]- mag_to_flambda_Vega(star['J'], 'J'))*100/flux[-1] > 1e2:
        AVs[0] = np.nan
    if abs(flux[-1]- mag_to_flambda_Vega(star['H'], 'H'))*100/flux[-1] > 1e2:
        AVs[1] = np.nan
    if abs(flux[-1]- mag_to_flambda_Vega(star['KS'], 'Ks'))*100/flux[-1] > 1e2:
        AVs[2] = np.nan

    AV = np.nanmean(AVs)
#    print(AV)
#    filtered_AV = [x for x in AVs if abs(x - AV) <= 5.0]
    
    return AV

def mag_to_flambda_Vega(mag, band):
    F_lambda_vega = {
            'B': 6.32e-9,
            'V': 3.64e-9,
            'J': 3.13e-10,
            'H': 1.13e-10,
            'Ks': 4.14e-11
        }
    return F_lambda_vega[band] * 10**(-0.4 * mag)


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
#
#        print(almaIII.columns.names)
#        for star in almaIII:
#            if '087' in star['HDMHDE']:
#                print(star['STARS'])

        # Load Gaia XP spectra
        with fits.open(crossmatch_file) as hdul:
            almaIII = hdul[1].data

#        for star in almaIII:
#            if '36 982' in star['HDMHDE']:
#                print(star['STARS'])
#        exit(0)

        with fits.open(almaII_file) as hdul:
            almaII = hdul[1].data

#        print(almaII.columns.names)
#        for star in almaII:
#            if '3326716470156761472' in str(star['GaiaDR2']):
#                print(star['ALS'], star['SpType'])
#
#        exit(0)

        # Load B and V filter curves
        B_filter = SpectralElement.from_file(f"{band_path}JohnB.dat")#, wave_unit='angstrom')
        V_filter = SpectralElement.from_file(f"{band_path}JohnV.dat")#, wave_unit='angstrom')

        # Loop over stars
        high_Rv_counter = 0
        low_Rv_counter = 0
        high_RV_stars = {}
        low_RV_stars = {}
        for star in almaIII:
            if star['STARS'] not in ['ALS 882',
                                     'HD 14 422',
                                     'HDE 237 056',
                                     'CPD -29 2176'
                                     'HDE 239 689',
                                     'HD 63 150',
                                     'HD 36 982',
                                     'theta^1 Ori BaBb',
                                     ]:
                continue
            wavelength = star['WAVE'] * 10  # Gaia XP: nm -> Å
            c_nm = 3e8 * 1e9

            # W/nm/m^2 = (1e7 erg/s) / (10 Å) / (1e4 cm^2 ) = 100 erg/s/cm²/Å
            flux = star['FLUX'] * 100 # W/nm/m^2 -> erg/s/cm²/Å
            flux_err = star['FLUX_ERR']

            # If spectra is empty skip star
            if sum(flux) == 0.: continue

            # Compute B, V synthetic magnitudes
            B_mag, V_mag, BminusV = synthetic_obs_BV(wavelength, flux, B_filter, V_filter)

            IImask = np.isin(almaII['GaiaDR2'], star['GAIA'])
            almaII_star = almaII[IImask]
            sptype = almaII_star['SpType'][0].split(', ')

            RVs = []
            for sp in sptype:
                BV0, JV0, HV0, KV0 = get_intrinsic_colors(sp)
                if BV0 is None:
                    continue

                E_BV = BminusV - BV0
                #E_VK = (V_mag - star['Ks']) - (-KV0)
                AV = calc_AV_JHKs(V_mag, star['J'], star['H'], star['KS'], JV0, HV0, KV0, flux)
                RV = np.array(AV) / E_BV
                RVs.append(RV)

            print(star['STARS'], sptype, RVs)#, E_BV, B_mag, V_mag)

            avg_RV = np.mean(RVs)

            band_x = np.array([4400, 5500,
                               12500, 16500, 21600])
            band_y = np.array([mag_to_flambda_Vega(B_mag, 'B'), mag_to_flambda_Vega(V_mag, 'V'),
                               mag_to_flambda_Vega(star['J'], 'J'), mag_to_flambda_Vega(star['H'], 'H'),
                               mag_to_flambda_Vega(star['KS'], 'Ks')
                               ])
#            if (avg_RV < 8.0) or E_BV < 0.3 or avg_RV < 1.0 or not np.isfinite(avg_RV): continue

            plt.plot(wavelength, flux-flux[0],
                     label = f"{star['STARS']}, RV ={avg_RV:.3f}")
            plt.scatter(band_x, band_y-flux[0]
                        )

            if (avg_RV > 5.0) and E_BV > 0.3 and avg_RV > 1.0: #or avg_RV < 2.5:
                high_Rv_counter += 1
                #print("high RV", ",", star['STARS'], ",", avg_RV)
                high_RV_stars[star['STARS']] = avg_RV

            if avg_RV < 2.5 and E_BV > 0.3 and avg_RV > 1.0:
                low_RV_of_star = avg_RV
                low_Rv_counter += 1
                #print("low RV", ",", star['STARS'], ",", avg_RV)
                low_RV_stars[star['STARS']] = avg_RV

        print(f"RESULTS:\nfound {low_Rv_counter} RV < 2.5 stars, {high_Rv_counter} RV > 5.0 stars.")

#        plt.yscale("log")
        plt.legend()
        plt.ylabel(r"flux (erg/s/cm$^2$/$\AA$)")
        plt.xlabel(r"$\lambda$ ($\AA$)")
        plt.show()

        exit(0)

        #outname = crossmatch_file.replace("_wsptype.fits", "_lowRV.fits")
        #low_RV_table = Table(rows=low_RV_stars, names=almaIII.columns.names)
        #low_RV_table.write(crossmatch_file, overwrite=True)
        #print(f"new file created: {crossmatch_file}")

        outname = crossmatch_file.replace("_wsptype.fits", "_extremeRV.fits")
        high_RV_table = Table(rows=high_RV_stars, names=almaIII.columns.names)
        high_RV_table.write(outname, overwrite=True)
        print(f"new file created: {outname}")
