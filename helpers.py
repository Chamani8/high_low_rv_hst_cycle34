import os
import glob
import numpy as np
import emcee
import h5py
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfWriter
from scipy.ndimage import gaussian_filter1d

from measure_extinction.modeldata import ModelData

def home_path():
    return os.path.expanduser("~")

def script_path():
    return os.path.dirname(os.path.abspath(__file__))

def datfile_path():
    return f"{home_path()}/extstar_data/DAT_files/"

def models_path():
    return f"{home_path()}/extstar_data/Models/"

def measure_extinction_path():
    return f"{home_path()}/measure_extinction/measure_extinction/"

def targetlist(sel_star=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    header = ""
    targetlist = {}
    with open(f"{script_dir}/target_list.dat", "r") as slist:
        for line in slist:
            if line[0]=="#":
                if header == "": 
                    header = line.split()
                    header[0] = header[0][1:]
                continue
            for head,data in zip(header, line.split()):
                if head == "spectral-type" and sel_star:
                    logTeff, Teff_std, logg, logg_std = conv_spectraltype(data)
                    pname = ["logTeff", "Teff_std", "logg", "logg_std"]
                    pval = [logTeff, Teff_std, logg, logg_std]
                elif head == "host-galaxy":
                    pname = ["logZ"]
                    if data == "LMC":
                        pval = [-0.35]
                    elif data == "SMC":
                        pval = [-0.65]
                    else:
                        pval = [0.]
                else:
                    try:
                        data = float(data)
                    except:
                        data = data
                    pname = [head]
                    pval = [data]
                
                for dict_name, dict_val in zip(pname, pval):
                    if dict_name not in targetlist.keys():
                        targetlist[dict_name] = [dict_val]
                    else:
                        targetlist[dict_name].append(dict_val)

    if sel_star:
        idx = targetlist["starname"].index(sel_star)
        return {key: values[idx] for key, values in targetlist.items()}
    else:
        return targetlist
    
def conv_spectraltype(star_spectral_type):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    Grav_const = 6.67E-11	# [ m^3 kg^-1 s^-2 ]  https://physics.nist.gov/cgi-bin/cuu/Value?bg
    Rsun = 6.96E+08	# [m] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    Msun = 1.99E+30	# [kg] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    default_logTeff = 4.25
    default_Teff_std = 0.2
    default_logg = 3.09
    default_logg_std = 0.5
    logTeff=0.
    logg=0.
    if star_spectral_type.startswith("O10"): 
        star_spectral_type.replace("O10", "B0")
    with open(f"{script_dir}/stellar_classification_table.dat", "r") as spec_table:
        for spec_dat in spec_table:
            spec_type = spec_dat.split()[0]
            if spec_type == star_spectral_type:
                logTeff = np.log10(float(spec_dat.split()[4]))
                spec_dat = [float(j) for j in spec_dat.split()[1:]]
                logg = np.log10(( Grav_const*float(spec_dat[0])*Msun / ((float(spec_dat[2])*Rsun)**2) )*100)

    # round up to one-two decimal point(s)
    if str(logg)[2] == "9":
        logg = float(str(f"{logg:0.2f}"))
    else:
        logg = float(str(f"{logg:0.1f}"))

    if logTeff==0. and logg==0.:
        #print(f"WARNING! COULD NOT FIND SPECTRAL TYPE {star_spectral_type}, USING DEFAULT")
        #print(f"logTeff = {default_logTeff} +/- {default_Teff_std}, logg = {default_logg} +/- {default_logg_std}")
        return default_logTeff, default_Teff_std, default_logg, default_logg_std
    else:
        Teff_std, logg_std = spectral_stddev(star_spectral_type)
        return logTeff, Teff_std, logg, logg_std

def spectral_stddev(star_spectral_type):
    # specs1, specs2 are for debugging purposes to see what spectral types
    # are used in the standard deviations of logTeff, logg respectively
    #
    # This function only works with the spectral table provided in the respository

    upper_spec, lower_spec = adj_Teff_spec(star_spectral_type)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temps = []
    specs1 = []
    loggs = []
    specs2 = []
    Grav_const = 6.67E-11	# [ m^3 kg^-1 s^-2 ]  https://physics.nist.gov/cgi-bin/cuu/Value?bg
    Rsun = 6.96E+08	# [m] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    Msun = 1.99E+30	# [kg] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    with open(f"{script_dir}/stellar_classification_table.dat", "r") as spec_table:
        prev_dat = "a a"
        for spec_dat in spec_table:
            spec_type = spec_dat.split()[0]
            if spec_type == upper_spec or spec_type == star_spectral_type or spec_type == lower_spec:
                specs1.append(spec_type)
                temps.append(np.log10(float(spec_dat.split()[4])))

            if spec_type == star_spectral_type:
                prev_dats = [float(j) for j in prev_dat.split()[1:]]
                logg = np.log10(( Grav_const*prev_dats[0]*Msun / ((prev_dats[2]*Rsun)**2) )*100)
                loggs.append(logg)
                specs2.append(prev_dat.split()[0])

                spec_dats = [float(j) for j in spec_dat.split()[1:]]
                logg = np.log10(( Grav_const*spec_dats[0]*Msun / ((spec_dats[2]*Rsun)**2) )*100)
                loggs.append(logg)
                specs2.append(spec_dat.split()[0])

            elif prev_dat.split()[0] == star_spectral_type:
                spec_dats = [float(j) for j in spec_dat.split()[1:]]
                logg = np.log10(( Grav_const*spec_dats[0]*Msun / ((spec_dats[2]*Rsun)**2) )*100)
                loggs.append(logg)
                specs2.append(spec_dat.split()[0])

            prev_dat = spec_dat

    return np.std(temps), np.std(loggs)

def adj_Teff_spec(spectral_type):
    romans = ['I', 'V', 'X']
    index2 = next((i for i, char in enumerate(spectral_type) if char in romans), -1)
    index3 = next((i for i, char in enumerate(spectral_type[index2:]) if char not in romans), -1)+index2
    spec_letter = spectral_type[0]
    spec_num = spectral_type[1:index2]
    spec_rom = spectral_type[index2:]
    spec_order = ["O", "B", "A", "F", "G", "K", "M", "L"]
    index = spec_order.index(spec_letter)

    if spectral_type[index2:index3] != "I":
        specn = float(spec_num)-1
        upper_spec = f"{spec_letter}{int(specn)}{spec_rom}"
        specn = float(spec_num)+1
        lower_spec = f"{spec_letter}{int(specn)}{spec_rom}"

        if spec_num == "0":
            upper_spec = spec_order[index - 1] if index > 0 else None
            upper_spec = f"{upper_spec}9{spec_rom}"
        elif spec_num == "9":
            lower_spec = spec_order[index + 1] if index < len(spec_order) - 1 else None
            lower_spec = f"{lower_spec}0{spec_rom}"
    else:
        specn = float(spec_num)-1
        upper_spec = f"{spec_letter}{int(specn)}VI"
        specn = float(spec_num)+1
        lower_spec = f"{spec_letter}{spec_num}II"

        if spec_num == "0":
            upper_spec = spec_order[index - 1] if index > 0 else None
            upper_spec = f"{upper_spec}9VI"
        if spec_num == "9":
            if spec_letter == "O":
                lower_spec = f"{spec_letter}{spec_num}II"
            else:
                next_letter = spec_order[index + 1] if index < len(spec_order) - 1 else None
                if next_letter:
                    lower_spec = f"{next_letter}0Ia0"

    return upper_spec, lower_spec

def read_h5(file,
            burnfrac=0.5,
            ):
    
    backend = emcee.backends.HDFBackend(file)
    #samples = backend.get_chain()
    nsteps = backend.iteration

    # Log probabilities
    #log_prob = backend.get_log_prob()
    #print(log_prob)

    # Autocorrelation time
    tau = backend.get_autocorr_time(tol=0)

    flat_samples = backend.get_chain(discard=int(burnfrac * nsteps), flat=True)
    p50 = np.percentile(flat_samples, 50, axis=0)
    params_per = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(flat_samples, [16, 50, 84], axis=0)),
        )
    n_params = len(p50)

    params_p50 = np.zeros(n_params)
    params_unc = np.zeros(n_params)
    for k, val in enumerate(params_per):
        params_p50[k] = val[0]
        params_unc[k] = 0.5 * (val[1] + val[2])

    return {"sampler": backend, "flat_samples": flat_samples, "tau": tau, "params_p50": params_p50, "params_unc": params_unc}

def pickle_modelfiles(picfilename, data_names, band_names, modstr = "tlusty_"):
    modpath = models_path()

    tlusty_models_fullpath = glob.glob(f"{modpath}{modstr}*.dat")
    tlusty_models = [
            tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
        ]

    if len(tlusty_models) > 1:
        print(f"{len(tlusty_models)} model files found.")
    else:
        raise ValueError("no model files found.")

    #print(data_names)
    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(
        tlusty_models,
        path=f"{modpath}/",
        band_names=band_names,
        spectra_names=data_names,
    )
    pickle.dump(modinfo, open(picfilename, "wb"))

def combine_images(file_match, file_ext, path, outpath=None):
    if not outpath: outpath = script_path()
    outname = file_match.replace("*", "combined")
    outname = outname.replace(".", f"_{file_ext}.")

    #gather list of images
    image_files = glob.glob(f"{path}/{file_match}")

    if file_match.endswith(".png"):
        #combine the images into one cumulative file
        image_list = []
        for png_file in image_files:
            img = Image.open(png_file)
            image_list.append(img.convert('RGB'))

        #check if a combined file already exists
        existing_file = glob.glob(f"{outpath}/{outname}")
        if len(existing_file) < 1:
            #save the new cumulative file with the name as is
            image_list[0].save(f"{outpath}/{outname}", save_all=True, append_images=image_list[1:])
            print("file written: ", f"{outpath}/{outname}")
        else:
            print("Warning! A combined file with the name ", f"{outpath}/{outname}", "already exists.")
            return
    
    elif file_match.endswith(".pdf"):
        #compile list of pdfs
        image_list = PdfWriter()
        for pdf in image_files:
            image_list.append(pdf)

        #check if a combined file already exists
        existing_file = glob.glob(f"{outpath}/{outname}")
        if len(existing_file) < 1:
            #save the new cumulative file with the name as is
            image_list.write(f"{outpath}/{outname}")
            image_list.close()
            print("file written: ", f"{outpath}/{outname}")
        else:
            print("Warning! A combined file with the name ", f"{outpath}/{outname}", "already exists.")
            return

def rebin_data(x, y, fwhm, factor=6): # fwhm units: micron
    stddev = fwhm/2.355 #fwhm/2sqrt(2*ln(2))
    delta_x = np.mean(np.diff(x.value))
    sigma_pix = stddev / delta_x
    y_out = gaussian_filter1d(y, sigma=sigma_pix)

    return x, y_out

if __name__ == "__main__":
    lmc_smc_targetlist = targetlist()
    print(lmc_smc_targetlist)
    print(home_path(), script_path(), 
          datfile_path(), models_path()
          )