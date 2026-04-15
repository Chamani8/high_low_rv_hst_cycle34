import os
import argparse
import glob
import pickle
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import argparse
from astropy.table import QTable

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.modeldata import ModelData
from measure_extinction.model import MEModel

import helpers

def calc_E4455(extdata):
    #if "STIS_Opt" not in extdata.waves.keys():
    #    return
    bindx = np.argsort(np.abs(extdata.waves["GAIA_BP"] - 0.44 * u.micron))[0]
    return (extdata.exts["GAIA_BP"][bindx], extdata.uncs["GAIA_BP"][bindx])


def create_extinction_curve(starname, best_param_fits = None, outname = f"_mefit_ext"):
    script_path = helpers.script_path()
    ext_folder = f"{script_path}/ext_curves"

    outname = f"{starname}{outname}.fits"
    if best_param_fits == None: 
        best_param_fits = f"{starname}_mefit_min_params.fits"

    # Read best-fit params from fits file
    try:
        qt = QTable.read(f"{script_path}/stellar_param_fits/minimizer/{best_param_fits}")
    except:
        print("Error: Could not find best-params file ", f"{script_path}/stellar_param_fits/minimizer/{best_param_fits}")
        return

    reddened_star = StarData(f"{starname}.dat", path=helpers.datfile_path(), only_bands=["J", "H", "K"]) #, only_data = ["STIS_Opt"]

    if "BAND" in reddened_star.data.keys():
        band_names = reddened_star.data["BAND"].get_band_names()
    else:
        band_names = []
    data_names = list(reddened_star.data.keys())

    picfilename = f"{script_path}/tlusty_gaia_bprp_modinfo.p"

    # Check if pickeled model exists in folder otherwise pickel the models
    try:
        modinfo = pickle.load(open(picfilename, "rb"))
    except:
        helpers.pickle_modelfiles(picfilename, data_names, band_names)
        print("Error: Re-run the script with newly created pickled file.")
        return

    fitmod = MEModel(modinfo=modinfo)
    # Copy the best fit params to memod
    for i,paramname in enumerate(qt["name"]):
        param = getattr(fitmod, paramname)
        param.value = qt["value"][i]
        param.unc = qt["unc"][i]
        param.fixed = qt["fixed"][i]

    # create a stardata object with the best intrinsic (no extinction) model
    modsed = fitmod.stellar_sed(modinfo)
    if "BAND" in reddened_star.data.keys():
        modinfo.band_names = reddened_star.data["BAND"].get_band_names()

    modsed_stardata = modinfo.SED_to_StarData(modsed)

    # create the extincion curve
    rel_band = 0.55*u.micron #"V"
    extdata = ExtData()
    extdata.calc_elx(reddened_star, modsed_stardata, rel_band=rel_band) # E(lambda - 55)

    # rather than using JHK to calculate E(B-V), R(V) and A(V), use the relative bad 0.55 microns
    extdata.columns["EBV"] = calc_E4455(extdata) # E(44-55)
    extdata.calc_AV_JHK() # A(55)   
    extdata.calc_RV() # R(55)

    EBV = extdata.columns["EBV"]
    AV = extdata.columns["AV"]
    RV = extdata.columns["RV"]
    print(f"{starname:<15}, {AV[0]:<10.3f}, {AV[1]:<10.3f}, {EBV[0]:<10.3f}, {EBV[1]:<10.3f}, {RV[0]:<10.3f}, {RV[1]:<10.3f}" )
    print(f"{script_path}RV_results.dat")
    with open(f'{script_path}/RV_results.dat', "a") as f:
        f.write(f"{starname:<15} {AV[0]:.3f} +/- {AV[1]:<10.3f} {EBV[0]:.3f} +/- {EBV[1]:<10.3f} {RV[0]:.3f} +/- {RV[1]:<10.3f}\n")

    # make sure the extinction curve folder exists
    os.makedirs(ext_folder, exist_ok=True)

    # save the extincion curve
    extdata.save(f"{ext_folder}/{outname}")#, fit_params=fit_params)
    print("Extinction curve saved in ", f"{ext_folder}/{outname}")


def main():
    parser = argparse.ArgumentParser(description="Select start to create extinction curve.")
    parser.add_argument('-s', '--starname', type=str, help="Target name", default=None)
    args = parser.parse_args()
    starname = args.starname

    # following lines are enabled for testing:
    #starname = "BE74-422"

    if starname == None:
        targetlist = helpers.targetlist()
        starnames = targetlist['starname']
        for starname in starnames:
            try:
                create_extinction_curve(starname.lower())
            except:
                continue
    else:
        create_extinction_curve(starname.lower())

if __name__ == "__main__":
    main()