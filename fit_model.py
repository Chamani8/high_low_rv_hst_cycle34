import os
import argparse
import helpers
import time
import glob
import pickle
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from measure_extinction.stardata import StarData
from measure_extinction.model import MEModel

from create_ext_curves import create_extinction_curve

def argument_parser():
    parser = argparse.ArgumentParser(description="Select start to fit and select to do mcmc fitting or not.")
    parser.add_argument('-s', '--starname', type=str, help="Target name", default=None)
    parser.add_argument('-m', '--run_mcmc', action="store_true", help="type of fitter")
    parser.add_argument('--nsteps', type=str, help="type of fitter", default=None)
    parser.add_argument('-p', '--showfit', action="store_true", help="Show plotted fitting result")
    parser.add_argument('--plot_h5', action="store_true", help="Plot fitting result from h5 file")
    parser.add_argument('--ignore_h5', action="store_true", help="Ignore existing h5 file.")
    args = parser.parse_args()

    return args.starname, args.run_mcmc, args.nsteps, args.showfit, args.plot_h5, args.ignore_h5

def parse_exclude_file(starname):
    exclude_list = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(f"{script_dir}/exclude_regions.dat", "r") as exclude_file:
            for region in exclude_file:
                if region[0] == "#": continue
                if region.split()[0].lower() == starname:
                    wave1 = float(region.split()[1])
                    wave2 = float(region.split()[2])
                    if wave1 > wave2:
                        exclude_list.append([1/wave1,1/wave2])
                    else:
                        exclude_list.append([1/wave2,1/wave1])
    except:
        exclude_list = []

    if exclude_list != []:
        return exclude_list/u.micron
    else:
        return None

def residual(fitmod,obsdata,modinfo):

    for cspec in obsdata.data.keys():
        modsed = fitmod.stellar_sed(modinfo)
        ext_modsed = fitmod.dust_extinguished_sed(modinfo, modsed)
        hi_ext_modsed = fitmod.hi_abs_sed(modinfo, ext_modsed)
        gvals = hi_ext_modsed[cspec] > 0.0

        modspec = hi_ext_modsed[cspec][gvals] * fitmod.norm.value
        uncs = obsdata.data[cspec].uncs.value[gvals] / modspec
        chisqr = ( obsdata.data[cspec].fluxes.value[gvals] - modspec ) ** 2 / (uncs**2)

    return np.sum(chisqr[np.isfinite(chisqr)])#, uncs


def fitmod_save_params(fitmod, path, filename):
        """
        Save the parameters with names and values
        """
        # line 1
        pnames = [
            ["logTeff", "logg", "logZ", "vturb", "velocity", "windamp", "windalpha"],
            ["Av", "Rv", "C2", "B3", "C4", "xo", "gamma"],
            ["vel_MW", "logHI_MW", "fore_Av", "fore_Rv", "vel_exgal", "logHI_exgal"],
        ]
        
        outlines=[]
        for cnames in pnames:
            hline = ""
            tline = ""
            for cname in cnames:
                if getattr(fitmod, cname).fixed:
                    fstr = "F"
                else:
                    fstr = ""
                if getattr(fitmod, cname).prior is not None:
                    fstr = f"{fstr}P"
                hline += f"{cname} "
                tline += f"{getattr(fitmod, cname).value:.3f}{fstr} "
            outlines.append(f"{tline[:-1]} ({hline[:-1]})\n")

        if hasattr(fitmod, "logf"):
            hline = "logf: "
            tline = ""
            for cname in fitmod.logf.keys():
                hline += f"{cname} "
                tline += f"{fitmod.logf[cname].value:.2f} "
            outlines.append(f"{tline[:-1]} ({hline[:-1]})\n")

        os.makedirs(path, exist_ok=True) # make path if it doesnt already exist
        with open(f"{path}/{filename}", "w") as f:
            f.write("best minimizer params\n")
            f.writelines(outlines)
        
        print(f"best parameters saved in {path}")


def fit_model(starname,
    run_mcmc = False,
    plot_h5 = False,
    ignore_h5 = False,
    nsteps = None,
    showfit = False,
    create_ext = True,
    only_data = ["GAIA_RP", "GAIA_BP"] # do 'None' to fit all available data UV + optical;
    ):

    print(f"Fitting {starname} ...")
    script_path = helpers.script_path()

    ### If h5 file with the given number steps already exists, skip fitting star
    h5_filename = f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit.h5"
    h5_files = glob.glob(h5_filename)

    h5_nsteps=0
    if h5_files:
        h5_file = helpers.read_h5(h5_filename)
        h5_nsteps = h5_file["sampler"].iteration

    print(starname, h5_nsteps)
    if nsteps == h5_nsteps and not ignore_h5:
        print(f"An HDF5 file already exists for {starname}. Aborting script. To run anyway, use command \'--ignore_h5\'.")
        return
    ####

    # Get priors as a dictionary
    priors = helpers.targetlist(starname)
    additional_exclude_reg = parse_exclude_file(starname)

    # How many times to weight photometry more?
    weight_photom_more = 5

    # Get star data only in the optical, and J, H, K photometry
    reddened_star = StarData(f"{starname.lower()}.dat", path=helpers.datfile_path(), only_data=only_data, only_bands=["J", "H", "K"]
                             )

#    print(reddened_star.data.keys(), helpers.datfile_path(), f"{starname.lower()}.dat")
#    return
    data_names = list(reddened_star.data.keys())
    # Retrieve the photometry if it exists
    if "BAND" in data_names:
        band_names = reddened_star.data["BAND"].get_band_names()
    else:
        band_names = []

    start_time = time.time()

    #if "GAIA_BPRP" in data_names:
    picfilename = f"{script_path}/tlusty_gaia_bprp_modinfo.p"

    try:
        modinfo = pickle.load(open(picfilename, "rb"))
    except:
        helpers.pickle_modelfiles(picfilename, data_names, band_names)
        print("Error: Re-run the script with newly created pickled file.")


    memod = MEModel(modinfo=modinfo)

    if "Teff" in reddened_star.model_params.keys():
        memod.logTeff.value = np.log10(float(reddened_star.model_params["Teff"]))
        memod.logTeff.fixed = True
    if "logg" in reddened_star.model_params.keys():
        memod.logg.value = float(reddened_star.model_params["logg"])
        memod.logg.fixed = True
    if "Z" in reddened_star.model_params.keys():
        memod.logZ.value = np.log10(float(reddened_star.model_params["Z"]))
        memod.logZ.fixed = True
    if "velocity" in reddened_star.model_params.keys():
        memod.velocity.value = float(reddened_star.model_params["velocity"])
        memod.velocity.fixed = True

    memod.add_exclude_region([1/0.3, 1/0.2]/u.micron)
    memod.add_exclude_region([1/0.672, 1/0.602]/u.micron)
    memod.add_exclude_region([1/1.017, 1/0.85]/u.micron)

    if additional_exclude_reg is not None:
        for cexreg in additional_exclude_reg:
            memod.add_exclude_region(cexreg)

    # add in 1% uncertainty on the STIS spectra
    gvals = reddened_star.data["GAIA_BP"].uncs > 0.0
    nuncs = (np.square(reddened_star.data["GAIA_BP"].uncs[gvals]) +
             np.square(reddened_star.data["GAIA_BP"].fluxes[gvals] * 0.01))
    reddened_star.data["GAIA_BP"].uncs[gvals] = np.sqrt(nuncs)

    gvals = reddened_star.data["GAIA_RP"].uncs > 0.0
    nuncs = (np.square(reddened_star.data["GAIA_RP"].uncs[gvals]) +
             np.square(reddened_star.data["GAIA_RP"].fluxes[gvals] * 0.01))
    reddened_star.data["GAIA_RP"].uncs[gvals] = np.sqrt(nuncs)

    memod.fit_weights(reddened_star)

    memod.weights["BAND"] *= weight_photom_more

    #fix_dust_params:
    memod.C2.fixed       = True
    memod.B3.fixed       = True
    memod.C4.fixed       = True
    memod.xo.fixed       = True
    memod.gamma.fixed    = True
    memod.vel_MW.fixed   = True
    memod.logHI_MW.fixed = True

    ## set priors ##
    memod.velocity.fixed = False
    if "velocity" in priors.keys():
        memod.velocity.value = priors["velocity"]
    else:
        memod.velocity.value = 0.0
    memod.velocity.prior = (0.0, 1.0)

    if "RV" in priors.keys():
        memod.Rv.value = priors["RV"]
    else:
        memod.Rv.value = 3.1
    memod.Rv.bounds = (1.0, 9.0)

    if "vturb" in priors.keys():
        memod.vturb.value = priors["vturb"]
    else:
        memod.vturb.value = 6.0

    if "AV" in priors.keys():
        memod.Av.value = priors["AV"]
    else:
        memod.Av.value = 1.0

    memod.logTeff.fixed = False
    memod.logTeff.value = priors["logTeff"]
    memod.logTeff.prior = (priors["logTeff"], priors["Teff_std"])
    memod.logTeff.bounds = (priors["logTeff"]-(priors["Teff_std"]*3),
                            priors["logTeff"]+(priors["Teff_std"]*3))

    memod.logg.fixed = False
    memod.logg.value = priors["logg"]
    memod.logg.prior = (priors["logg"], priors["logg_std"])

    #if "logZ" in priors.keys():
    memod.logZ.value = 0.0#priors["logZ"]
    memod.logZ.fixed = True # fix logZ == 0.0 for MW

    memod.set_initial_norm(reddened_star, modinfo)

    fit_params = {}

    print("initial parameters")
    memod.pprint_parameters()
    start_time = time.time()

    fitmod, result = memod.fit_minimizer(reddened_star, modinfo, maxiter=20000)

    # Minimizer results
    chisqr = residual(fitmod, reddened_star, modinfo)
    print(f"chi^2 = {chisqr}")
    print(f"Finished optinizer fit in {(time.time() - start_time)} seconds")
    print(result["message"])
    if result["message"] != "Optimization terminated successfully.":
        print("Aborting fit ...")
        return
    print("best parameters")
    fitmod.pprint_parameters()
    fitmod_save_params(fitmod, f"{script_path}/stellar_param_fits/minimizer/", f"{starname.lower()}_mefit_min_params.dat")
    fitmod.save_parameters(filename=f"{script_path}/stellar_param_fits/minimizer/{starname.lower()}_mefit_min_params.fits")
    fitmod.plot(reddened_star, modinfo)
    plt.savefig(f"{script_path}/stellar_param_fits/minimizer/{starname.lower()}_mefit_min.pdf")
    plt.close()

    if run_mcmc:
        os.makedirs(f"{script_path}/stellar_param_fits/mcmc/", exist_ok=True)
        fitmod2, flat_samples, sampler = fitmod.fit_sampler(
            reddened_star,
            modinfo,
            nsteps=nsteps,
            save_samples=h5_filename,
        )

        print("p50 parameters")
        fitmod2.pprint_parameters()
        fitmod_save_params(fitmod2, f"{script_path}/stellar_param_fits/mcmc/", f"{starname.lower()}_mefit_params.dat")
        fitmod.save_parameters(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_params.fits")

        fitmod2.plot(reddened_star, modinfo)
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc.pdf")
        plt.close()

        fitmod2.plot_sampler_chains(sampler)
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc_chains.pdf")
        plt.close()

        fitmod2.plot_sampler_corner(flat_samples)
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc_corner.pdf")
        plt.close()

    elif plot_h5:
        #TODO: add a check for if the h5 file exists
        h5_file = helpers.read_h5(h5_filename)
        #print(f"Autocorrelation times: {h5_file["tau"]}")
        #print(f"p50 best-fit: {h5_file["params_p50"]}")
        #print(f"p50 best-fit: {h5_file["params_p50"]}")
        fitmod.fit_to_parameters(h5_file["params_p50"], uncs=h5_file["params_unc"])

        fitmod.plot(reddened_star, modinfo)
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc.pdf")
        plt.close()

        fitmod.plot_sampler_corner(h5_file["flat_samples"])
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc_chains.pdf")
        plt.close()

        fitmod.plot_sampler_chains(h5_file["sampler"])
        plt.savefig(f"{script_path}/stellar_param_fits/mcmc/{starname.lower()}_mefit_mcmc_corner.pdf")
        plt.close()
    
    if showfit:
        fitmod.plot(reddened_star, modinfo)
        plt.show()

    if create_ext:
        create_extinction_curve(starname.lower())


def main():
    starname, run_mcmc, nsteps, showfit, plot_h5, ignore_h5 = argument_parser()

    # following lines are enabled for testing:
    # TODO: at the end comment out following lines
    if starname == None: starname = "ALS612"
    if run_mcmc == None: run_mcmc = False 
    #if nsteps == None: nsteps = 200000

    if starname == None:
        targetlist = helpers.targetlist()
        starnames = targetlist['starname']
        for starname in starnames:
            fit_model(starname,
                      run_mcmc=False,
                      nsteps=nsteps,
                      showfit=showfit,
                      create_ext=True,
                      plot_h5=plot_h5,
                      ignore_h5=ignore_h5)

    else:
        fit_model(starname,
                  run_mcmc=False,
                  nsteps=nsteps, 
                  showfit=True,
                  create_ext=True,
                  plot_h5=plot_h5,
                  ignore_h5=ignore_h5)

    # TODO: at the end comment out following line
    helpers.combine_images("*_mefit_min.pdf", "15Apr26", 
                           f"{helpers.script_path()}/stellar_param_fits/minimizer")

if __name__ == "__main__":
    main()

    #TODO:
    # determine correct nsteps based on autocorrelation time