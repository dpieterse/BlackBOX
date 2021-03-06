
from zogy import *

import re, fnmatch
from itertools import chain

from astropy.coordinates import Angle
from astropy.table import vstack

# trouble installing fitsio in singularity container so create
# fallback on astropy.io.fits (imported in zogy import above)
try:
    import fitsio
    use_fitsio = True
except:
    use_fitsio = False

from multiprocessing import Pool, Manager, Lock, Queue, Array

import aplpy

from blackbox import date2mjd, str2bool, unzip
from blackbox import get_par, already_exists, copy_files2keep
from blackbox import create_log, define_sections, fpack
from qc import qc_check, run_qc_check

import set_zogy
import set_buildref as set_br
import set_blackbox as set_bb


__version__ = '0.4'


################################################################################

def buildref (telescope=None, date_start=None, date_end=None, field_ID=None,
              filters=None, qc_flag_max=None, seeing_max=None,
              make_colfig=False, filters_colfig='iqu'):
    
    
    """Module to consider one specific or all available field IDs within
    a specified time range, and to combine the available images of
    that field ID in one or all filters, using those images that
    satisfy the quality flag and seeing constraint. The combining of
    the images is done using the function imcombine.
    
    The resulting reference image is put through zogy.py as the
    reference image and the corresponding reference directory is
    prepared.


    To do: 
    ------
    
    (15) when something is wrong with a single file (e.g. mini
         background image not present), prevent imcombine from
         stopping

    (18) edges of images can be noisy compared to the rest of the image;
         could be improved by considering those pixels with less than
         certain amount of images (e.g. 3) as edge pixels.

    (19) reduce size of bkg and bkg_std images in output directory
         by creating mini images; N.B.: currently the function mini2back
         uses the background box size from the zogy settings file - it 
         would be better to use the value saved in the header of the mini
         images, as the background box size with which a mini image was 
         made may not be the same as the currently set value in the 
         settings file.

    Done:
    -----

    (1) the background in the resulting reference image is around
        zero, while zogy.py is presently setup to compute the image noise
        characteristics using the background level. Possible solutions:

        - add articial background to the reference image, where the level
          is determined by the variance of the background in the image
          and the effective read noise:

            bkg level = bkg variance - (effective readnoise)**2
   
          pro: no need to update zogy.py
          con: need to determine background again in zogy.py, which
               in principle is not needed

        + rewrite zogy such that scatter in the background is used to
          estimate the noise, rather than the background level and
          readnoise**2; e.g. in optimal photometry

          pro: would be good to have this option of zero background
               which would also save time as background subtraction
               does not need to be redone
          con: work to do, need to be careful

          in this case, should add "BKG_SUB" keyword with boolean value so
          that background subtraction can be skipped in zogy.py

    (2) check if all new reference image keywords are added (see
        Overleaf document) and that effective readnoise is correctly
        determined - latter is not so important anymore is the
        image background variance is used for the noise estimation
        (see item 1)
    
    (3) add option to delete temporary directories

    (4) add option to correct images for gain correction factors
        (GAINCF1-16) that are recorded in the header of the master flat 
        (whose name is listed in reduced image header keyword MFLAT-F)

    (5) the main reference image (which combination method: weighted mean?
        weighted clipped mean?) should have the same background subtracted
        as was done for the individual images (and resulting catalogs) and
        that should be put through zogy.                                  
    
        for the moment, add additional reference images that can be
        compared with; no need to compute the masks or zogy products
        for these:

        - using a bigger background box to avoid subtraction of
          small-scale structures
        - alternative combination methods

    (6) add wildcard options in field_ID, so e.g. 16??? can 
        be used to process all fields >= 16000

    (7) when zogy is run on reference image that is already background
        subtracted, no need to do so again; through new header keyword
        BKG_SUB T/F? - now done by producing ..bkg_std.fits (inferred
        from the combined weights image) and .._bkg.fits, which is just
        a zero-valued image.

        Related: if SWarp WEIGHT_TYPE input MAP_RMS is used instead
        of MAP_WEIGHT, that output weights image is also RMS; this could
        then be used directly as the output .._bkg_std.fits of the 
        reference image, while ..bkg.fits could be a zero-valued image.
        - this MAP_RMS option doesn't seem to work: results in 
        intermediate resampled weights images and combined weights image
        with mainly zeros.

        Need to find out if flux scaling factor p (see Eqs. 26 and 27
        of SWarp manual) is applied to the weights image in SWarp 
        internally or not. Could be done by running a test with/without
        FSCALE header keyword, and see if anything changes in output
        weights - found out: SWarp applies this p internally also to
        the weights, so input weights need to be defined without p:
        w = 1/bkg_std**2.

    (8) add option to use the field center as defined in the grid
        besides e.g. first image, last image, median image position

    (9) combine build_ref_MLBG and imcombine_MLBG into single module
        (buildref?) to be placed in ZOGY directory, and add its settings
        file (set_buildref?)

    (10) add fpacking at the end (and probably turn off fpacking of
         reference images in blackbox, to avoid simultaneous fpacking of
         the same files)

    (11) add buildref and set_buildref to singularity container

    (12) check if flux scaling is ok; need to include airmass? - 
         a good option is to scale the flux of all images to as if
         the image was taken at airmass=1 (see paragraph below)

         related: avoid calculating airmass for reference image that
         is a combination of various images, as this will lead to wrong
         values due to the DATE-OBS being an average value. Best to
         scale the reference image fluxes to airmass=1 and change the
         airmass values in the reference header. 
         Then when photometric calibration is performed, it should use
         the header airmass value rather than to recalculate it. If some
         images are taking at high airmass, the standard deviation in the
         zeropoint determination of the reference image will slightly
         increase, because the airmass changes signficantly across the
         field. E.g. at an airmass of 2, the change in airmass across the
         field is about 0.1 airmass, which would lead to an increased
         scatter of about 0.1*0.5=0.05mag in the u filter, but less than 
         about 0.02 for the other filters - seems acceptable. Could
         provide a warning about this if an image airmass is high.

         Alternative (correct) solution: determine scaling due to
         airmass as a function of pixel position, i.e. a 2D airmass
         map. Could determine this by calculating the airmass for
         a (random?) set of limited points across the image and 
         apply interpolation (scipy.interpolate.interpn or 
         scipy.interpolate.griddata)

    (13) switch from dfits to reading header using astropy

    (14) handle cases when fits and/or its mask is not fpacked
         (could be the case when blackbox/zogy are funpacking
          while the refbuild module is running)
    
         related: include both fpacked and funpacked files; right
         before they are read, check which one to use

    (16) why is S-SEEING keyword missing in so many images? - there
         was a bug in v0.9.1 of BlackBOX/ZOGY that the headers of the
         reduced images were not updated if there was a red flag inside
         ZOGY; moreover, the headers of the catalogs were also not updated
         with the flags occurring inside ZOGY; fixed in v0.9.2.

    (17) add two settings parameters that determine the subset of
         images in case of (too) many images of the same field
         in the same filter:
         - number of best images to select
         - header keyword on which to sort the best images, e.g.
           LIMMAG or S-SEEING

    """
    

    global tel, q, genlog, lock, max_qc_flag, max_seeing, start_date, end_date
    global time_refstart
    tel = telescope
    lock = Lock()
    max_qc_flag = qc_flag_max
    max_seeing = seeing_max
    start_date = date_start
    end_date = date_end

    # record starting time to add to header
    time_refstart = Time.now().isot
    
    # initialize logging
    if not os.path.isdir(get_par(set_bb.log_dir,tel)):
        os.makedirs(get_par(set_bb.log_dir,tel))

    q = Manager().Queue() #create queue for logging
    genlogfile = '{}/{}_{}_buildref.log'.format(get_par(set_bb.log_dir,tel), tel,
                                                Time.now().strftime('%Y%m%d_%H%M%S'))
    genlog = create_log (genlogfile)

    
    q.put(genlog.info('building reference images'))
    q.put(genlog.info('log file: {}'.format(genlogfile)))
    q.put(genlog.info('number of processes: {}'.format(get_par(set_br.nproc,tel))))
    q.put(genlog.info('number of threads: {}'.format(get_par(set_br.nthread,tel))))


    t0 = time.time()
    

    # prepare a table with filenames and relevant header keywords
    # -----------------------------------------------------------
    
    red_path = get_par(set_bb.red_dir,tel)
    filenames = glob.glob('{}/*/*/*/*_red.fits*'.format(red_path))
    q.put(genlog.info('total number of files: {}'
                      .format(len(filenames))))
    
    # split into [nproc] lists
    nfiles = len(filenames)
    nproc = get_par(set_br.nproc,tel)
    list_of_filelists = []
    index = np.linspace(0,nfiles,num=nproc+1).astype(int)
    for i in range(nproc):
        list_of_filelists.append(filenames[index[i]:index[i+1]])
        

    # use function pool_func_alt to read headers from files
    # using multiple processes and write them to a table
    try:
        results = pool_func_alt (header2table, list_of_filelists)
        # stack separate tables in results
        table = vstack(results)
    except Exception as e:
        q.put(genlog.error (traceback.format_exc()))
        q.put(genlog.error ('exception was raised during [pool_func_alt]: {}'
                            .format(e)))
        raise RuntimeError


    q.put(genlog.info('number of files with all required keywords: {}'
                      .format(len(table))))
    q.put(genlog.info('file headers read in {:.2f}s'.format(time.time()-t0)))

    
    # filter table entries based on date, field_ID, filter, qc-flag and seeing
    # ------------------------------------------------------------------------
    
    # function to set start and end dates
    def set_date (date, start=True):
        
        """helper function to set start/end dates"""
        
        # if no date is specified, include all data from 10 years ago
        # until now
        if date is None:
            mjd = mjd_now
            if start:
                mjd -= 365.25 * 10
        else:
            # if specific date is specified, convert it to mjd
            if len(date) < 8:
                mjd = mjd_now + round(float(date))
            # otherwise assume date is relative wrt now/today
            else:
                date = re.sub(',|-|\.|\/', '', date)
                mjd = date2mjd ('{}'.format(date), time_str='12:00')

        return mjd

    mjd_now = int(Time.now().mjd) + 0.5
    mjd_start = set_date (date_start)
    mjd_end = set_date (date_end, start=False)
    # select relevant table entries
    mask = ((table['MJD-OBS'] >= mjd_start) & (table['MJD-OBS'] <= mjd_end))
    table = table[mask]
    q.put(genlog.info('number of files left (date_start/end cut): {}'
                      .format(len(table))))
    

    # if object (field ID) is specified, which can include the unix
    # wildcards * and ?, select only images with a matching object
    # number
    if field_ID is not None:
        mask = [fnmatch.fnmatch(str(obj), field_ID) for obj in table['OBJECT']]
        table = table[mask]
        q.put(genlog.info('number of files left (FIELD_ID cut): {}'
                          .format(len(table))))


    # if filter(s) is specified, select only images with filter(s)
    # specified
    if filters is not None:
        #mask = [table['FILTER'][i] in filters for i in range(len(table))]
        mask = [filt in filters for filt in table['FILTER']]
        table = table[mask]
        q.put(genlog.info('number of files left (FILTER cut): {}'
                          .format(len(table))))
        
        
    # if qc_flag_max is specified, select only images with QC-FLAG of
    # qc_flag_max and better
    if len(table)>0 and qc_flag_max is not None:
        qc_col = ['green', 'yellow', 'orange', 'red']
        # redefine qc_col up to and including qc_flag_max
        qc_col = qc_col[0:qc_col.index(qc_flag_max)+1]

        mask_green = [table['QC-FLAG'][i].strip()=='green' for i in range(len(table))]
        mask_yellow = [table['QC-FLAG'][i].strip()=='yellow' for i in range(len(table))]
        mask_orange = [table['QC-FLAG'][i].strip()=='orange' for i in range(len(table))]
        mask_red = [table['QC-FLAG'][i].strip()=='red' for i in range(len(table))]
        q.put(genlog.info('number of green: {}, yellow: {}, orange: {}, red: {}'
                          .format(np.sum(mask_green), np.sum(mask_yellow),
                                  np.sum(mask_orange), np.sum(mask_red))))
        
        # strip table color from spaces
        mask = [table['QC-FLAG'][i].strip() in qc_col for i in range(len(table))]
        table = table[mask]
        q.put(genlog.info('number of files left (QC-FLAG cut): {}'
                          .format(len(table))))


    # if max_seeing is specified, select only images with the same or
    # better seeing
    if max_seeing is not None:
        mask = (table['S-SEEING'] <= max_seeing)
        table = table[mask]
        q.put(genlog.info('number of files left (SEEING cut): {}'
                          .format(len(table))))


    # for table entries that have survived the cuts, prepare the list
    # of imagelists with the accompanying lists of field_IDs and
    # filters
    objs_uniq = np.unique(table['OBJECT'])
    filts_uniq = np.unique(table['FILTER'])
    list_of_imagelists = []
    obj_list = []
    filt_list = []
    for obj in objs_uniq:
    
        # skip fields '00000' and those beyond 20,000
        if int(obj) == 0 or int(obj) >= 20000:
            continue

        for filt in filts_uniq:
            mask = ((table['OBJECT'] == obj) & (table['FILTER'] == filt))
            ntrue = np.sum(mask)
            if ntrue > 1:
                nmax = get_par(set_br.subset_nmax,tel)
                if ntrue > nmax:
                    # more images available than [set_br.subset_nmax];
                    # select subset of images, sorted by header
                    # keyword also defined in settings file
                    key_sort = get_par(set_br.subset_key,tel)
                    indices_sort = np.argsort(table[key_sort][mask])
                    # reverse indices if high end values should be
                    # included
                    if not get_par(set_br.subset_lowend,tel):
                        indices_sort = indices_sort[::-1]
                    # threshold value
                    threshold = table[key_sort][mask][indices_sort[nmax]]
                    print ('threshold: {}'.format(threshold))
                    # update mask
                    if get_par(set_br.subset_lowend,tel):
                        mask &= (table[key_sort] < threshold)
                    else:
                        mask &= (table[key_sort] > threshold)
                    q.put(genlog.info ('selected following subset of {} images '
                                       'for field_ID {}, filter: {}, based on '
                                       'header keyword: {}:\n{}'
                                       .format(nmax, obj, filt, key_sort,
                                               table[mask])))
                    
                # add this set of images with their field_ID and
                # filter to the lists of images, field_IDs and filters
                # to be processed
                list_of_imagelists.append(list(table['FILE'][mask]))
                obj_list.append(table['OBJECT'][mask][0])
                filt_list.append(table['FILTER'][mask][0])
                

    if len(table)==0:
        q.put(genlog.warning ('zero field IDs with sufficient number of '
                              'good images to process'))
        logging.shutdown()
        return


    # multiprocess remaining files
    # ----------------------------

    # feed the lists that were created above to the multiprocessing
    # helper function [pool_func_alt] that will arrange each
    # process to call [prep_ref] to prepare the reference image
    # for a particular field and filter combination, using
    # the [imcombine] function 
    try:
        result = pool_func_alt (prep_ref, list_of_imagelists, obj_list,
                                filt_list)
    except Exception as e:
        q.put(genlog.error (traceback.format_exc()))
        q.put(genlog.error ('exception was raised during [pool_func_alt]: {}'
                            .format(e)))
        raise RuntimeError


    # make color figures
    # ------------------

    if make_colfig:

        q.put(genlog.info ('preparing color figures'))

        # also prepare color figures
        try:
            result = pool_func_copy (prep_colfig, objs_uniq, filters_colfig)
        except Exception as e:
            q.put(genlog.error (traceback.format_exc()))
            q.put(genlog.error ('exception was raised during [pool_func_copy]: {}'
                                .format(e)))
            raise RuntimeError

        
    # fpack reference fits files
    # --------------------------

    ref_dir = get_par(set_bb.ref_dir,tel)
    list_2pack = []
    for field_ID in objs_uniq:
        ref_path = '{}/{:05}'.format(ref_dir, field_ID)
        ref_path = '{}_alt'.format(ref_path)
        list_2pack.append(glob.glob('{}/*.fits'.format(ref_path)))

    # unnest nested list_2pack
    list_2pack = list(chain.from_iterable(list_2pack))
    q.put(genlog.info ('list of images to fpack: {}'.format(list_2pack)))

    # use [pool_func_copy] to process the list
    result = pool_func_copy (fpack_copy, list_2pack)

    
    logging.shutdown()
    return
    
    
################################################################################

def fpack_copy (filename):

    """Fpack fits images; skip fits tables"""

    try:
    
        # fits check if extension is .fits and not an LDAC fits file
        if filename.split('.')[-1] == 'fits' and '_ldac.fits' not in filename:
            header = read_hdulist(filename, get_data=False, get_header=True,
                                  ext_name_indices=0)

            # check if it is an image
            if header['NAXIS']==2:
                # determine if integer or float image
                if header['BITPIX'] > 0:
                    cmd = ['fpack', '-D', '-Y', '-v', filename]
                else:
                    if 'Scorr' in filename or 'limmag' in filename:
                        quant = 1
                    else:
                        quant = 16
                    cmd = ['fpack', '-q', str(quant), '-D', '-Y', '-v', filename]
                subprocess.call(cmd)

    except Exception as e:
        q.put(genlog.info(traceback.format_exc()))
        q.put(genlog.error('exception was raised in fpacking of image {} {}'
                           .format(filename,e)))


################################################################################

def pool_func_copy (func, filelist, *args):

    try:
        results = []
        pool = Pool(get_par(set_br.nproc,tel))
        for filename in filelist:
            args_temp = [filename]
            for arg in args:
                args_temp.append(arg)
            results.append(pool.apply_async(func, args_temp))

        pool.close()
        pool.join()
        results = [r.get() for r in results]
        #q.put(logger.info('result from pool.apply_async: {}'.format(results)))
    except Exception as e:
        q.put(genlog.info(traceback.format_exc()))
        q.put(genlog.error('exception was raised during [pool.apply_async({})]: {}'
                           .format(func, e)))
        raise SystemExit
        

################################################################################

def pool_func_alt (func, list_of_imagelists, *args):
    
    try:
        results = []
        pool = Pool(get_par(set_br.nproc,tel))
        for nlist, filelist in enumerate(list_of_imagelists):
            args_temp = [filelist]
            for arg in args:
                args_temp.append(arg[nlist])
                
            results.append(pool.apply_async(func, args_temp))

        pool.close()
        pool.join()
        results = [r.get() for r in results]
        #q.put(genlog.info('result from pool.apply_async: {}'.format(results)))
        return results
        
    except Exception as e:
        q.put(genlog.info(traceback.format_exc()))
        q.put(genlog.error('exception was raised during [pool.apply_async({})]: {}'
                          .format(func, e)))
        raise RuntimeError
    

################################################################################

def header2table (filenames):

    # initialize rows
    rows = []

    # keywords to add to table
    keys = ['MJD-OBS', 'OBJECT', 'FILTER', 'QC-FLAG',
            # add keyword that is sorted on if many images are available
            get_par(set_br.subset_key,tel)]
    if max_seeing is not None:
        keys.append('S-SEEING')

    # loop input list of filenames
    for filename in filenames:

        # check if filename exists, fpacked or not
        exists, filename = already_exists (filename, get_filename=True)
        if exists:
            if use_fitsio:
                # read header; use fitsio as it is faster than
                # astropy.io.fits on when not using solid state disk
                with fitsio.FITS(filename) as hdulist:
                    h = hdulist[-1].read_header()
            else:
                with fits.open(filename) as hdulist:
                    h = hdulist[-1].read_header()
                    
        else:
            q.put(genlog.warning('file does not exist; skipping image {}'
                                 .format(filename)))
            continue

                
        # check if all keywords present before appending to table
        append = True
        for key in keys:
            if key not in h:
                q.put(genlog.warning('keyword {} not in header; skipping image {}'
                                     .format(key, filename)))
                append = False
                break

        if False:
            # up to v0.9.2, it was possible for the reduced image
            # header not to contain a red flag, while the catalog file
            # did contain it - this happened when zogy was not
            # processed properly (e.g. astrometry.net failed), then
            # dummy catalogs were produced but the reduced image
            # header was not updated - this bug was fixed in v0.9.2.
            if '.fz' in filename:
                catname = filename.replace('.fits.fz', '_cat.fits')
            else:
                catname = filename.replace('.fits', '_cat.fits')
            # if it doesn't exist, continue with the next
            if not os.path.isfile(catname):
                q.put(genlog.warning('catalog file {} does not exist; skipping image {}'
                                     .format(catname, filename)))
                continue

            # read header
            if use_fitsio:
                with fitsio.FITS(catname) as hdulist:
                    h_cat = hdulist[-1].read_header()
            else:
                with fits.open(catname) as hdulist:
                    h_cat = hdulist[-1].read_header()

            # if a dummycats were created, copy the qc-flag to the reduced
            # image header
            if h_cat['DUMMYCAT']:
                key = 'QC-FLAG'
                h[key] = h_cat[key]
                    
                
        if append:
            # prepare row of filename and header values
            row = [filename]
            for key in keys:
                row.append(h[key])
            # append to rows
            rows.append(row)

            
    # create table from rows
    names = ['FILE']
    for key in keys:
        names.append(key)

    dtypes = ['S', float, int, 'S1', 'S6', float]
    if max_seeing is not None:
        dtypes.append(float)

    if len(rows) == 0:
        # rows without entries: create empty table
        table = Table(names=names, dtype=dtypes)
    else: 
        table = Table(rows=rows, names=names, dtype=dtypes)

        
    return table


################################################################################

def prep_colfig (field_ID, filters):
    
    # determine reference directory and file
    ref_path = '{}/{:05}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    # for the moment, add _alt to this path to separate it from
    # existing reference images
    ref_path = '{}_alt'.format(ref_path)

    # header keyword to use for scaling (e.g. PC-ZP or LIMMAG)
    key = 'LIMMAG'
    
    # initialize rgb list of images
    images_rgb = []
    images_std = []
    images_zp = []
    
    for filt in filters:

        image = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)
        exists, image = already_exists(image, get_filename=True)
        
        if not exists:
            q.put (genlog.info('{} does not exist; not able to prepare color '
                               'figure for field_ID {}'.format(image, field_ID)))
            return
        else:
            
            # add to image_rgb list (unzip if needed)
            image = unzip(image, put_lock=False)
            images_rgb.append(image)
            
            # read image data and header
            data, header = read_hdulist(image, get_header=True)
            
            # determine image standard deviation
            mean, median, std = sigma_clipped_stats(data)
            images_std.append(std)
            
            # read header zeropoint
            if key in header:
                images_zp.append(header[key])
            else:
                q.put (genlog.info('missing header keyword {}; not able to '
                                   'prepare color figure for field_ID {}'
                                   .format(key, field_ID)))
                return
            
    # scaling
    f_min = 0
    vmin_r = f_min * images_std[0]
    vmin_g = f_min * images_std[1]
    vmin_b = f_min * images_std[2]

    f_max = 10
    vmax_r = f_max * images_std[0] * 10**(-0.4*(images_zp[2]-images_zp[0]))
    vmax_g = f_max * images_std[1] * 10**(-0.4*(images_zp[2]-images_zp[1]))
    vmax_b = f_max * images_std[2]

    # make color figure
    colfig = '{}/{}_{}_{}.png'.format(ref_path, tel, field_ID, filters)
    aplpy.make_rgb_image(images_rgb, colfig,
                         vmin_r=vmin_r, vmax_r=vmax_r,
                         vmin_g=vmin_g, vmax_g=vmax_g,
                         vmin_b=vmin_b, vmax_b=vmax_b)
    

    
################################################################################

def prep_ref (imagelist, field_ID, filt):
    
    # determine reference directory and file
    ref_path = '{}/{:05}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    # for the moment, add _alt to this path to separate it from
    # existing reference images
    ref_path = '{}_alt'.format(ref_path)
    
    make_dir (ref_path)
    ref_fits_out = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)
    
    
    # if reference image already exists, check if images used are the
    # same as the input [imagelist]
    exists, ref_fits_temp = already_exists (ref_fits_out, get_filename=True)
    if exists:
        q.put (genlog.info('reference image {} already exists; checking if it '
                          'needs updating'.format(ref_fits_out)))
        # read header
        header_ref = read_hdulist (ref_fits_temp, get_data=False, get_header=True)
        # check how many images were used
        if 'R-NUSED' in header_ref:
            n_used = header_ref['R-NUSED']
        else:
            n_used = 1

        # gather used images into list
        if 'R-IM1' in header_ref:
            imagelist_used = [header_ref['R-IM{}'.format(i+1)]
                              for i in range(n_used)
                              if 'R-IM{}'.format(i+1) in header_ref]

        # compare input [imagelist] with [imagelist_used]; if they are
        # the same, no need to build this particular reference image
        # again
        imagelist_new = [image.split('/')[-1].split('.fits')[0]
                         for image in imagelist]
        if set(imagelist_new) == set(imagelist_used):
            # same sets of images, return
            q.put (genlog.info ('imagelist_new: {}'.format(imagelist_new)))
            q.put (genlog.info ('imagelist_used: {}'.format(imagelist_used)))
            q.put (genlog.info ('reference image with same set of images already present; skipping'))
            return

        
    # prepare temporary folder
    # for the moment, add _alt to this path to separate it from
    # existing reference images
    tmp_path = '{}/{:05}_alt/{}'.format(get_par(set_bb.tmp_dir,tel), field_ID,
                                    ref_fits_out.split('/')[-1].replace('.fits',''))
    make_dir (tmp_path, empty=True)
    
    # names of output fits and its mask
    ref_fits = '{}/{}'.format(tmp_path, ref_fits_out.split('/')[-1])
    ref_fits_mask = ref_fits.replace('red.fits','mask.fits')

    # create logfile specific to this reference image in tmp folder
    # (to be copied to final output folder at the end)
    logfile = ref_fits.replace('.fits', '.log')
    log = create_log (logfile)

    # run imcombine
    log.info('running imcombine; outputfile: {}'.format(ref_fits))

    try: 
        imcombine (field_ID,
                   imagelist,
                   ref_fits,
                   get_par(set_br.combine_type,tel),
                   masktype_discard = get_par(set_br.masktype_discard,tel),
                   tempdir = tmp_path,
                   center_type = get_par(set_br.center_type,tel),
                   back_type = get_par(set_br.back_type,tel),
                   back_size = get_par(set_br.back_size,tel),
                   back_filtersize = get_par(set_br.back_filtersize,tel),
                   swarp_cfg = get_par(set_zogy.swarp_cfg,tel),
                   nthreads = get_par(set_br.nthread,tel),
                   log=log)

    except Exception as e:
        q.put (genlog.info (traceback.format_exc()))
        q.put (genlog.error ('exception was raised during [imcombine]: {}'
                            .format(e)))
        log.info (traceback.format_exc())
        log.error ('exception was raised during [imcombine]: {}'.format(e))
        raise RuntimeError
                
        
    # run zogy on newly prepared reference image
    try:
        zogy_processed = False
        header_optsub = optimal_subtraction(
            ref_fits=ref_fits, ref_fits_mask=ref_fits_mask,
            set_file='set_zogy', log=log, verbose=None,
            nthread=get_par(set_br.nthread,tel), telescope=tel)
    except Exception as e:
        q.put (genlog.info(traceback.format_exc()))
        q.put (genlog.error('exception was raised during reference [optimal_subtraction]: {}'
                           .format(e)))
        log.info (traceback.format_exc())
        log.error ('exception was raised during reference [optimal_subtraction]: {}'
                   .format(e))

    else:
        zogy_processed = True

    finally:
        if not zogy_processed:
            q.put (genlog.error('due to exception: returning without copying reference files'))
            log.error ('due to exception: returning without copying reference files')
            return

    log.info('zogy_processed: {}'.format(zogy_processed))


    # copy/move files to the reference folder
    tmp_base = ref_fits.split('_red.fits')[0]
    ref_base = ref_fits_out.split('_red.fits')[0]
    # remove old reference files
    oldfiles = glob.glob('{}*'.format(ref_base))
    if False:
        for f in oldfiles:
            os.remove(f)
    else:
        # or move them to an Old folder instead
        old_path = '{}/Old/'.format(ref_path)
        make_dir (old_path)
        for f in oldfiles:
            f_dst = '{}/{}'.format(old_path,f.split('/')[-1])
            shutil.move (f, f_dst)
        
    # now move [ref_2keep] to the reference directory
    result = copy_files2keep(tmp_base, ref_base, get_par(set_bb.ref_2keep,tel),
                             move=False, log=log)
    # include full background and background standard deviation images
    bkg_2keep = ['_bkg.fits', '_bkg_std.fits']
    result = copy_files2keep(tmp_base, ref_base, bkg_2keep, move=False, log=log)
    

    # also build a couple of alternative reference images for
    # comparison; name these ...._whatever_red.fits, so that they do
    # get copied over to the reference folder below (which uses the
    # file extensions defined in blackbox settings file)
    if True:
        center_type = get_par(set_br.center_type,tel)
        masktype_discard = get_par(set_br.masktype_discard,tel)
        
        def help_imcombine (combine_type, back_type, back_default=0,
                            back_size=30, back_filtersize=5):

            if back_type == 'auto':
                ext = '_{}_{}_{}_{}.fits'.format(combine_type, back_type,
                                                 back_size, back_filtersize)
            elif back_type == 'manual':
                ext = '_{}_{}_{}.fits'.format(combine_type, back_type,
                                              back_default)
            elif back_type == 'constant':
                ext = '_{}_{}_clipmed.fits'.format(combine_type, back_type)
            else:
                ext = '_{}_{}.fits'.format(combine_type, back_type)

            ref_fits_temp = ref_fits.replace('.fits', ext)   

            imcombine (field_ID,
                       imagelist,
                       ref_fits_temp,
                       combine_type,
                       back_type=back_type,
                       back_default=back_default,
                       back_size=back_size,
                       back_filtersize=back_filtersize,
                       center_type=center_type,
                       masktype_discard=masktype_discard,
                       tempdir=tmp_path,
                       remap_each=False,
                       swarp_cfg=get_par(set_zogy.swarp_cfg,tel),
                       nthreads=get_par(set_br.nthread,tel),
                       log=log)

            # copy combined image to reference folder
            shutil.move (ref_fits_temp, ref_path)

            
        if False:
            help_imcombine ('weighted', 'blackbox')
            help_imcombine ('clipped', 'auto', back_size=60, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=120, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=240, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=960, back_filtersize=5)
            help_imcombine ('average', 'none')
            help_imcombine ('clipped', 'constant')



    # delete [tmp_path] if [set_br.keep_tmp] not True
    if not get_par(set_br.keep_tmp,tel) and os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


    log.info('finished making reference image: {}'.format(ref_fits_out))

    if False:
        
        # feed [header_optsub] to [run_qc_check] and check for a red flag
        qc_flag = run_qc_check (header_optsub, tel, log=log)
        qc_flag = 'green'
    
        if qc_flag != 'red':
            tmp_base = ref_fits.split('.fits')[0]
            # now move [ref_2keep] to the reference directory
            ref_base = ref_fits_out.split('.fits')[0]
            result = copy_files2keep(tmp_base, ref_base,
                                     get_par(set_bb.ref_2keep,tel),
                                     move=False, log=log)
            log.info('finished making reference image: {}'.format(ref_fits_out))

        else:
            log.info('encountered red flag; not using image: {} as reference'
                     .format(ref_fits))
            
        
################################################################################
            
def make_dir(path, empty=False):

    """Wrapper function to lock make_dir_nolock so that it's only used by
       1 process. """

    lock.acquire()
    make_dir_nolock (path, empty)
    lock.release()

    return


################################################################################

def make_dir_nolock(path, empty=False):

    """Function to make directory. If [empty] is True and the directory
       already exists, it will first be removed.
    """

    # if already exists but needs to be empty, remove it first
    if os.path.isdir(path) and empty:
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)

    return


################################################################################

def tune_gain (data, header):

    # master flat name from header
    mflat = header['MFLAT-F']

    # extract yyyymmdd from name
    mdate = mflat.split('_')[1]

    # infer full path to the master flat and its read header
    red_path = get_par(set_bb.red_dir,tel)
    mpath = '{}/{}/{}/{}/flat'.format(red_path, mdate[0:4], mdate[4:6], mdate[6:8])
    header_master = read_hdulist ('{}/{}'.format(mpath, mflat),
                                  get_data=False, get_header=True)
    
    # infer the data sections and number of channels
    __, __, __, __, data_sec_red = define_sections(data.shape)
    nchans = np.shape(data_sec_red)[0]

    # loop channels and apply the gain correction factors
    for i_chan in range(nchans):
        factor =  header_master['GAINCF{}'.format(i_chan)]
        data[data_sec_red[i_chan]] *= header_master[factor]

        log.info('gain tuned with correction factor {} for channel {}'
                 .format(factor, i_chan))

        
    return data


################################################################################

def imcombine (field_ID, imagelist, outputfile, combine_type, overwrite=True,
               masktype_discard=None, tempdir='.temp', center_type='first',
               use_wcs_center=True, back_type='auto', back_default=0,
               back_size=120, back_filtersize=3, resample_suffix='_resamp.fits',
               remap_each=True, remap_suffix='_remap.fits', swarp_cfg=None,
               nthreads=0, log=None):


    """Module to combine MeerLICHT/BlackGEM images.  The headers of the
    input images (all assumed to be useable, i.e. no red flags) need
    to have a WCS solution that SWarp understands, as SWarp is used to
    project the images to a common WCS frame, before the combining of
    the separate images is done.

    If the input images have an accompanying mask, i.e. with the same
    base name and containing "mask", then that mask will be used to
    avoid using e.g. edge pixels or cosmic rays in the combination.

    To do:

    - include 3 main parts:
      --> basic combination, including weights (using SWarp)
      --> optimal combination (ZOGY)
      --> astrodrizzle

    - filter out qc-flag==red images

    - include photometric scaling (FSCALE_DEFAULT) using the zeropoint
      ZP, airmass A and extinction coefficient k, to re-calculate
      the Fn/Fr flux ratio (header Z-FNR is often not available):

                 Fn/Fr = 10**(ZPn-ZPr-k*(An-Ar))

      (should Airmass be included??)

    """

    
    t0 = time.time()

    # check if there are at least a couple of images to combine
    if len(imagelist) < 2:
        raise RuntimeError ('too few images ({}) selected'.format(len(imagelist)))
    else:
        log.info ('{} images selected to combine'.format(len(imagelist)))
        
        
    if os.path.isfile(outputfile) and not overwrite:
        raise RuntimeError ('output image {} already exist'
                            .format(outputfile))

    # if outputmask already exists, raise error
    outputmask = outputfile.replace('red.fits', 'mask.fits')
    if os.path.isfile(outputfile) and not overwrite:
        raise RuntimeError ('output image {} already exist'
                            .format(outputfile))
    
    # if output background standard deviation image (=sqrt(1/weights
    # image)) already exists, raise error
    output_bkg_std = outputfile.replace('.fits', '_bkg_std.fits')
    if os.path.isfile(output_bkg_std) and not overwrite:
        raise RuntimeError ('output background STD image {} already exist'
                            .format(output_bkg_std))
    
    
    # clean up or make temporary directory if it is not the current directory '.'
    if tempdir[-1]=='/':
        tempdir = tempdir[0:-1]
    if tempdir != '.':
        if os.path.isdir(tempdir):
            if os.listdir(tempdir):
                log.info ('cleaning temporary directory {}'.format(tempdir))
                cmd = 'rm {}/*'.format(tempdir)
                result = subprocess.call(cmd, shell=True)
        else:
            cmd = ['mkdir','{}'.format(tempdir)]
            result = subprocess.call(cmd)


    # if SWarp configuration file does not exist, create default one in [tempdir]
    if swarp_cfg is None:
        swarp_cfg = tempdir+'/swarp.config'
        cmd = 'swarp -d > {}'.format(swarp_cfg)
        result = subprocess.call(cmd, shell=True)
    else:
        if not os.path.isfile(swarp_cfg):
            raise IOError ('file {} does not exist'.format(swarp_cfg))

        
    # COMBINE TYPE MEDIAN
    # Tells SWarp how to combine resampled images:
    # MEDIAN Take the median of pixel values
    # AVERAGE Take the average
    # MIN Take the minimum
    # MAX Take the maximum
    # WEIGHTED Take the weighted average
    # CHI2 Take the weighted, quadratic sum
    # SUM Take the sum
    # not in latest manual v2.21 (code is at v2.38)
    # CLIPPED, CHI-OLD, CHI-MODE, CHI-MEAN, WEIGHTED_WEIGHT, MEDIAN_WEIGHT,
    # AND, NAND, OR or NOR

    # make sure combine_type, back_type and center_type are lower case
    combine_type = combine_type.lower()
    back_type = back_type.lower()
    center_type = center_type.lower()
    
    # check if value of [combine_type] is valid; if not, exit
    combine_type_list = ['median', 'average', 'min', 'max', 'weighted', 'chi2', 'sum',
                         'clipped', 'weighted_weight', 'median_weight']
    if combine_type not in combine_type_list:
        raise ValueError ('[combine_type] method "{}" should be one of {}'.
                          format(combine_type, combine_type_list))


    # initialize arrays to keep
    nimages = len(imagelist)
    ra_centers = np.zeros(nimages)
    dec_centers = np.zeros(nimages)
    xsizes = np.zeros(nimages, dtype=int)
    ysizes = np.zeros(nimages, dtype=int)
    zps = np.zeros(nimages)
    airmasses = np.zeros(nimages)
    gains = np.zeros(nimages)
    rdnoises = np.zeros(nimages)
    saturates = np.zeros(nimages)
    exptimes = np.zeros(nimages)
    mjds = np.zeros(nimages)
    weights_mean = np.zeros(nimages)
    
    
    # initialize image_names that refer to fits images in [tempdir]
    image_names = np.array([])
    weights_names = np.array([])
    if remap_each:
        mask_names = np.array([])


    # loop input list of images
    for nimage, image in enumerate(imagelist):
        
        if not os.path.isfile(image):
            raise RuntimeError ('input image {} does not exist'.format(image))

        # read input image data and header
        data, header = read_hdulist(image, get_header=True)

        # if set_br.tune_gain switch is set to True, apply gain
        # correction factors inferred from master flat - switched
        # this off as the images have already been corrected
        # for this additional gain factor by flatfielding
        #if get_par(set_br.tune_gain,tel):
        #    data = tune_gain (data, header)
            
        # read corresponding mask image
        image_mask = image.replace('red.fits', 'mask.fits')
        data_mask, header_mask = read_hdulist(image_mask, get_header=True,
                                              dtype='uint8')

        # read corresponding mini background image
        image_bkg_mini = image.replace('red.fits', 'red_bkg_mini.fits')
        data_bkg_mini, header_bkg_mini = read_hdulist(image_bkg_mini,
                                                      get_header=True,
                                                      dtype='float32')
        # convert mini to full background image
        bkg_boxsize = header_bkg_mini['BKG_SIZE']
        data_bkg = mini2back (data_bkg_mini, data.shape, log,
                              order_interp=2,
                              bkg_boxsize=bkg_boxsize, timing=False)

        # read mini background standard deviation image
        image_bkg_std_mini = image.replace('red.fits', 'red_bkg_std_mini.fits')
        data_bkg_std_mini, header_bkg_std_mini = read_hdulist(image_bkg_std_mini,
                                                              get_header=True,
                                                              dtype='float32')

        # convert mini to full background standard deviation image
        bkg_boxsize = header_bkg_std_mini['BKG_SIZE']
        data_bkg_std = mini2back (data_bkg_std_mini, data.shape, log,
                                  order_interp=1,
                                  bkg_boxsize=bkg_boxsize, timing=False)
        
        # read relevant header keywords
        keywords = ['naxis1', 'naxis2', 'ra', 'dec', 'pc-zp', 'pc-zpstd',
                    'airmass', 'pc-extco', 'gain', 'rdnoise', 'saturate',
                    'exptime', 'mjd-obs']
        try:
            xsize, ysize, ra_temp, dec_temp, zp, zp_std, airmass, extco, gain, \
                rdnoise, saturate, exptime, mjd_obs = read_header_alt (header,
                                                                       keywords)
        except Exception as e:
            log.warning('exception was raised when reading header of image {}\n'
                        'not using it in image combination'.format(image, e))
            continue

        
        # determine weights image (1/variance) 
        # for Poisson noise component, use background image instead of
        # image itself:
        #data_var = data_bkg + rdnoise**2
        data_var = data_bkg_std**2
        data_weights = data_var
        index_nonzero = np.nonzero(data_var)
        data_weights[index_nonzero] = 1./data_var[index_nonzero]

        if False:
            # alternatively, provide the absolute values of the
            # background RMS map and using WEIGHT_TYPE MAP_RMS below;
            # however: this results in the resampled weights maps
            # (except for the one of the very first image) and also
            # the output weights map to contain mainly zeros
            data_weights = np.abs(data_bkg_std)


        # set pixels in data_mask that are to be discarded (selected
        # with input parameter masktype_discard) to zero in weights image
        mask_weights = np.zeros(data_mask.shape, dtype=bool)
        mask_value = set_zogy.mask_value
        # iterate over all mask values
        for val in mask_value.values():
            # check if this one is to be discarded
            if masktype_discard & val == val:
                mask_discard = (data_mask & val == val)
                mask_weights[mask_discard] = True
                log.info('discarding mask value {}; no. of pixels: {}'
                         .format(val, np.sum(mask_discard)))
                

        # set corresponding pixels to zero in data_weights
        data_weights[mask_weights] = 0

        # fix pixels
        data = fixpix (data, data_bkg, log, satlevel=saturate,
                       data_mask=data_mask, mask_value=mask_value)
        
        # fill arrays with header info
        xsizes[nimage] = xsize
        ysizes[nimage] = ysize
        zps[nimage] = zp
        airmasses[nimage] = airmass
        gains[nimage] = gain
        rdnoises[nimage] = rdnoise
        saturates[nimage] = saturate
        exptimes[nimage] = exptime
        mjds[nimage] = mjd_obs
        
        # calculate flux ratio (fscale in SWarp speak) using the
        # zeropoint difference between image and first image
        # (i.e. scale the fluxes in image to match those in the first
        # image), and scale all images to an airmass of 1
        # N.B.: this will lead to the reference image having the same
        # zeropoint as the first image, which may be confusing; could
        # also scale it to the image with highest zp, but then need
        # to do a separate loop inferring the zps of all images first,
        # but that is not that straightforward
        dmag = zp - zps[0] - extco*(airmass - 1)
        fscale = 10**(dmag/-2.5)

        # add fscale to image header
        header['FSCALE'] = (fscale, 'flux ratio wrt to first image and at airmass=1')
        
        # update weights image with scale factor according to Eq. 26
        # or 27 in SWarp manual:
        # N.B.: this is done internally by SWarp!!!
        #data_weights /= fscale**2

        # mean image weight
        weights_mean[nimage] = np.median(data_weights)
        
        
        # determine and record image centers
        if use_wcs_center:
            # determine WCS center of field
            wcs = WCS(header)
            ra_temp, dec_temp = wcs.all_pix2world(int(xsize/2), int(ysize/2), 1)
            #print ('center RA, DEC', ra_temp, dec_temp)
            #print ('center RA, DEC', Angle(ra_temp, unit=u.deg).to_string(u.hour),
            #       Angle(dec_temp, unit=u.deg).to_string(u.deg))

        else:
            # alternatively, using header keywords read above
            if ':' in ra_temp:
                ra_temp = Angle(ra_temp, unit=u.hour).degree
            if ':' in dec_temp:
                dec_temp = Angle(dec_temp, unit=u.degree).degree
            
        ra_centers[nimage] = ra_temp
        dec_centers[nimage] = dec_temp


        # save image in temp folder; first subtract background if
        # background option 'blackbox' is selected
        if back_type == 'blackbox':
            data -= data_bkg
            # set edge pixel values of image to zero, otherwise those
            # pixels will be negative and will/may end up in the edge
            # of the combined image
            data[data_mask==mask_value['edge']] = 0

        elif back_type == 'constant':
            # subtract one single value from the image: clipped median
            data_mean, data_median, data_std = sigma_clipped_stats (data)
            data -= data_median
            # make sure edge pixels are zero
            data[data_mask==mask_value['edge']] = 0

            
        image_temp = tempdir+'/'+image.split('/')[-1].replace('.fz','')
        fits.writeto(image_temp, data, header=header, overwrite=True)
        # add to array of names
        image_names = np.append(image_names, image_temp)


        # save weights image in the temp folder
        weights_temp = image_temp.replace('.fits','_weights.fits')
        fits.writeto(weights_temp, data_weights, overwrite=True)
        # add to array of weights names
        weights_names = np.append(weights_names, weights_temp)


        if remap_each:
            # save mask image in temp folder
            mask_temp = image_temp.replace('red.fits', 'mask.fits')
            fits.writeto(mask_temp, data_mask, header=header_mask, overwrite=True)
            # add to array of names
            mask_names = np.append(mask_names, mask_temp)
        

    # for the new reference image, adopt the size of the first image
    # for the moment; could expand that to a bigger image while using
    # [center_type]=all in SWarp
    refimage_xsize = xsizes[0]
    refimage_ysize = ysizes[0]
    size_str = str(refimage_xsize) + ',' + str(refimage_ysize)
    
    if center_type == 'first':
        ra_center = ra_centers[0]
        dec_center = dec_centers[0]
    elif center_type == 'last':
        ra_center = ra_centers[-1]
        dec_center = dec_centers[-1]
    elif center_type == 'mean':
        ra_center = np.mean(ra_centers)
        dec_center = np.mean(dec_centers)
    elif center_type == 'median':
        ra_center = np.median(ra_centers)
        dec_center = np.median(dec_centers)
    elif center_type == 'grid':
        # read from grid definition file located in ${ZOGYHOME}/CalFiles
        mlbg_fieldIDs = get_par(set_bb.mlbg_fieldIDs,tel)
        table_ID = ascii.read(mlbg_fieldIDs, names=['ID', 'RA', 'DEC'], data_start=0)
        # check if there is a match with the defined field ID
        mask_match = (table_ID['ID']==int(field_ID))
        if sum(mask_match) == 0:
            # observed field is not present in definition of field IDs
            msg = ('input field ID not present in definition of field IDs:\n{}\n'
                   'header field ID: {}\nnot processing'
                   .format(mlbg_fieldIDs, field_ID))
            q.put(genlog.error(msg))
            log.error(msg)
            return
        else:
            i_ID = np.nonzero(mask_match)[0][0]
            ra_center = table_ID['RA'][i_ID]
            dec_center = table_ID['DEC'][i_ID]
            log.info('adopting grid coordinates: ra_center: {}, dec_center: {}'
                     .format(ra_center, dec_center))
            
    else:
        raise ValueError ('input [center_type] not one of '
                          '[first, last, mean, median, grid]')

    # convert coordinates to input string for SWarp
    radec_str = '{},{}'.format(ra_center, dec_center)


    # set background settings in SWarp; if input background option was
    # 'blackbox', the background was already subtracted from the image
    if back_type == 'blackbox' or back_type == 'constant' or back_type == 'none':
        subtract_back_SWarp = 'N'
        back_type_SWarp = 'manual'
    else:
        subtract_back_SWarp = 'Y'
        back_type_SWarp = back_type

    # keywords to copy from original image to reference
    keys2copy = ['BUNIT', 'XBINNING', 'YBINNING',
                 'RADESYS', 'EPOCH',
                 'OBJECT', 'IMAGETYP', 'FILTER', 'EXPTIME',
                 'SITELAT', 'SITELONG', 'ELEVATIO',
                 'CCD-ID', 'CONTROLL', 'DETSPEED', 
                 'CCD-NW', 'CCD-NH',
                 'ORIGIN', 'TELESCOP', 'INSTRUME', 
                 'OBSERVER', 'ABOTVER']

    # run SWarp
    cmd = ['swarp', ','.join(image_names),
           '-c', swarp_cfg,
           '-COMBINE', 'Y',
           '-COMBINE_TYPE', combine_type.upper(),
           #'-WEIGHT_IMAGE', ','.join(weights_names),
           '-WEIGHT_SUFFIX', '_weights.fits',
           '-WEIGHTOUT_NAME', output_bkg_std,
           '-WEIGHT_TYPE', 'MAP_WEIGHT',
           '-RESCALE_WEIGHTS', 'N',
           '-CENTER_TYPE', 'MANUAL',
           '-CENTER', radec_str,
           '-IMAGE_SIZE', size_str,
           '-IMAGEOUT_NAME', outputfile,
           '-RESAMPLE_DIR', tempdir,
           '-RESAMPLE_SUFFIX', resample_suffix,
           '-RESAMPLING_TYPE', 'LANCZOS3',
           # GAIN_KEYWORD cannot be GAIN, as the value of GAIN1 is then adopted
           '-GAIN_KEYWORD', 'whatever',
           '-GAIN_DEFAULT', '1.0',
           '-SATLEV_KEYWORD', get_par(set_zogy.key_satlevel,tel),
           '-SUBTRACT_BACK', subtract_back_SWarp,
           '-BACK_TYPE', back_type_SWarp.upper(),
           '-BACK_DEFAULT', str(back_default),
           '-BACK_SIZE', str(back_size),
           '-BACK_FILTERSIZE', str(back_filtersize),
           '-FSCALE_KEYWORD', 'FSCALE',
           '-FSCALE_DEFAULT', '1.0',
           '-FSCALASTRO_TYPE', 'FIXED',
           '-VERBOSE_TYPE', 'FULL',
           '-NTHREADS', str(nthreads),
           '-COPY_KEYWORDS', ','.join(keys2copy),
           '-WRITE_FILEINFO', 'Y',
           '-WRITE_XML', 'N',
           '-VMEM_DIR', '.',
           '-VMEM_MAX', str(4096),
           '-MEM_MAX', str(4096),
           '-DELETE_TMPFILES', 'N',
           '-NOPENFILES_MAX', '256']


    cmd_str = ' '.join(cmd)
    log.info ('executing SWarp command:\n{}'.format(cmd_str))
    result = subprocess.call(cmd)
        
    # update header of outputfile
    data_out, header_out = read_hdulist(outputfile, get_header=True)

    # with RA and DEC
    header_out['RA'] = (ra_center, '[deg] telescope right ascension')
    header_out['DEC'] = (dec_center, '[deg] telescope declination')

    # with gain, readnoise, saturation level, exptime and mjd-obs
    gain, rdnoise, saturate, exptime, mjd = calc_headers (
        combine_type, gains, rdnoises, saturates, exptimes, mjds)
    
    if False:
        header_out['GAIN'] = (gain, '[e-/ADU] effective gain')
        header_out['RDNOISE'] = (rdnoise, '[e-] effective read-out noise')
        header_out['SATURATE'] = (saturate, '[e-] effective saturation threshold')
        header_out['EXPTIME'] = (exptime, '[s] effective exposure time')
        header_out['DATE-OBS'] = (Time(mjd, format='mjd').isot, 'average date of observation')
        header_out['MJD-OBS'] = (mjd, '[days] average MJD')
    else:
        date_obs = Time(mjd, format='mjd').isot
        header_out.set('GAIN', gain, '[e-/ADU] effective gain', after='DEC')
        header_out.set('RDNOISE', rdnoise, '[e-] effective read-out noise', after='GAIN')
        header_out.set('SATURATE', saturate, '[e-] effective saturation threshold', after='RDNOISE')
        header_out.set('EXPTIME', exptime, '[s] effective exposure time', after='SATURATE')
        header_out.set('DATE-OBS', date_obs, 'average date of observation', after='EXPTIME')
        header_out.set('MJD-OBS', mjd, '[days] average MJD', after='DATE-OBS')

    
    # buildref version
    header_out['R-V'] = (__version__, 'reference building module version used')

    # time when module was started
    header_out['R-TSTART'] = (time_refstart, 'UT time that module was started')
    
    # number of images used
    header_out['R-NUSED'] = (len(imagelist), 'number of images used to combine')
    # names of images that were used
    for nimage, image in enumerate(image_names):
        image = image.split('/')[-1].split('.fits')[0]
        header_out['R-IM{}'.format(nimage+1)] = (image, 'image {} used to combine'
                                                 .format(nimage+1))

    # combination method
    header_out['R-COMB-M'] = (combine_type,
                              'input images combination method')
    # background subtraction method
    header_out['R-BKG-M'] = (back_type,
                             'input images background subtraction method')
    # background subtracted?
    if back_type == 'none':
        bkg_sub = False
    else:
        bkg_sub = True

    header_out['BKG-SUB'] = (bkg_sub, 'sky background was subtracted?')
    
    # centering method
    header_out['R-CNTR-M'] = (center_type,
                              'reference image centering method')
    # discarded mask values
    header_out['R-MSKREJ'] = (masktype_discard,
                              'reject pixels with mask values part of this sum')
    
    val_str = '[{},{}]'.format(start_date, end_date)
    header_out['R-TRANGE'] = (val_str,
                              '[date/days] use images <= these limits of R-TSTART')
    
    header_out['R-QCMAX'] = (max_qc_flag, 'use images <= this QC flag')
    
    header_out['R-SEEMAX'] = (max_seeing, '[arcsec] use images <= this seeing')

    
    # any nan value in the image?
    mask_infnan = ~np.isfinite(data_out)
    if np.any(mask_infnan):
        log.info ('combined image contains non-finite numbers; replace with 0')
        data_out[mask_infnan] = 0


    # fluxes of individual images were scaled to airmass=1, and set
    # header AIRMASS accordingly
    header_out['AIRMASS'] = (1.0, 'Airmass forced to 1 in refbuild module')
    

    # time stamp of writing file
    ut_now = Time.now().isot
    header_out['DATEFILE'] = (ut_now, 'UTC date of writing file')
    header_out['R-DATE'] = (ut_now, 'time stamp reference image creation')
    # write file
    fits.writeto(outputfile, data_out.astype('float32'), header_out, overwrite=True)
    

    # replace output combined weights image with background standard
    # deviation values
    data_weights, header_weights = read_hdulist(output_bkg_std, get_header=True)
    mask_nonzero = (data_weights != 0)
    data_bkg_std = data_weights
    data_bkg_std[mask_nonzero] = 1./np.sqrt(data_weights[mask_nonzero])
    # replace zeros with median
    data_bkg_std[~mask_nonzero] = np.median(data_bkg_std[mask_nonzero])

    # write file
    header_weights['COMMENT'] = 'combined weights image was converted to STD image: std=1/sqrt(w)'
    header_weights['DATEFILE'] = (ut_now, 'UTC date of writing file')
    fits.writeto(output_bkg_std, data_bkg_std.astype('float32'), header_weights,
                 overwrite=True)

    # if background was subtracted, write zero-valued background image
    # for use in zogy
    if bkg_sub:
        output_bkg = outputfile.replace('.fits', '_bkg.fits')
        data_bkg = np.zeros(data_bkg_std.shape, dtype='float32')
        fits.writeto(output_bkg, data_bkg, overwrite=True)

        
    
    if remap_each:
        # median image weight
        weights_mean[nimage] = np.median(data_weights)

        log.info ('remapping individual images')
        
        # also SWarp individual images, e.g. for colour combination
        refimage = outputfile
        header_refimage = read_hdulist(refimage, get_data=False, get_header=True)

        # initialize combined mask
        mask_array_shape = (len(mask_names), refimage_ysize, refimage_xsize)
        data_mask_array = np.zeros(mask_array_shape, dtype='uint8')
        
        for nimage, image in enumerate(image_names):

            # skip remapping of images themselves for the moment; only
            # needed if some combination of the images other than
            # those available in SWarp is needed
            if False:

                t_temp = time.time()
                image_remap = image.replace('.fits', remap_suffix)
                
                log.info ('refimage: {}'.format(refimage))
                log.info ('image: {}'.format(image))
                log.info ('image_remap: {}'.format(image_remap))
                
                if not os.path.isfile(image_remap):
                    try:
                        result = run_remap_local (refimage, image, image_remap,
                                                  [refimage_ysize,refimage_xsize],
                                                  1.0, log, config=swarp_cfg,
                                                  resample='N', resample_dir=tempdir,
                                                  resample_suffix=resample_suffix,
                                                  nthreads=nthreads)
                    except Exception as e:
                        log.error (traceback.format_exc())
                        log.error ('exception was raised during [run_remap]: {}'
                                   .format(e))
                        raise RuntimeError
                    else:
                        log.info ('time spent in run_remap: {}'
                                  .format(time.time()-t_temp))
                        

            # same for image masks if there are any
            if len(mask_names) >= nimage:

                image_mask = mask_names[nimage]
                
                log.info ('processing mask: {}'.format(image_mask))
                
                # first need to update header of mask with header
                # of corresponding image
                header_image = read_hdulist(image, get_data=False, get_header=True)
                data_mask, header_mask = read_hdulist(image_mask, get_header=True)
                header = header_mask + header_image
                fits.writeto(image_mask, data_mask, header=header, overwrite=True)

                t_temp = time.time()
                image_mask_remap = image_mask.replace('.fits', remap_suffix) 
                if not os.path.isfile(image_mask_remap):

                    try:
                        result = run_remap_local (refimage, image_mask, image_mask_remap,
                                                  [refimage_ysize,refimage_xsize],
                                                  gain=1.0, log=log,
                                                  config=swarp_cfg,
                                                  resampling_type='NEAREST',
                                                  resample_dir=tempdir,
                                                  resample_suffix=resample_suffix,
                                                  dtype=data_mask.dtype.name,
                                                  value_edge=32,
                                                  nthreads=nthreads,
                                                  oversampling=0)
                                                  
                    except Exception as e:
                        log.error (traceback.format_exc())
                        log.error ('exception was raised during [run_remap]: {}'
                                   .format(e))
                        raise RuntimeError
                    else:
                        log.info ('wall-time spent in remapping mask: {}'
                                  .format(time.time()-t_temp))


                t_temp = time.time()

                # alternative way for masks, or images where
                # interpolation is not really needed by simply mapping
                # values from mask image to the refimage using header WCS
                # info
                if False:
                    def trans_func (output_coords):
                        ra_temp, dec_temp = wcs_mask.all_pix2world(
                            output_coords[0], output_coords[1], 0)
                        x, y = wcs_ref.all_world2pix(ra_temp, dec_temp, 0)
                        return (x,y) 
                    
                    global wcs_mask, wcs_ref
                    wcs_mask = WCS(header)
                    wcs_ref = WCS(header_refimage)
                    data_mask_remap_alt = ndimage.geometric_transform(data_mask, trans_func)

                    # write to image
                    fits.writeto(image_mask.replace('.fits','_alt'+remap_suffix),
                                 data_mask_remap_alt)
                    
                    log.info ('wall-time spent in remapping alt mask: {}'
                              .format(time.time()-t_temp))
                    
                
                # SWarp has converted input mask to float32, so need to
                # read fits image back into integer array to use in
                # combination of masks below
                data_mask_remap = (read_hdulist(image_mask_remap,get_header=False)
                                   +0.5).astype('uint8')

                # perform bitwise OR combination of mask_remap
                if nimage==0:
                    data_mask_OR = data_mask_remap
                else:
                    data_mask_OR = data_mask_OR | data_mask_remap

                # save mask in 3D array
                data_mask_array[nimage] = data_mask_remap

                
        # combine mask, starting from the OR combination of all masks
        data_mask_comb = data_mask_OR

        # but if at least 3 pixels are not masked, i.e. equal to zero,
        # then set combined mask also to zero
        mask_array_zeros = (data_mask_array == 0)
        mask_array_zerosum = np.sum(mask_array_zeros, axis=0)
        log.info ('np.shape(mask_array_zerosum): {}'
                  .format(np.shape(mask_array_zerosum)))
        mask_zero = (mask_array_zerosum >= 3)
        data_mask_comb[mask_zero] = 0
                
        # write combined mask to fits image 
        fits.writeto(outputmask, data_mask_comb, overwrite=overwrite)
        
        # feed resampled images to function [buildref_optimal]
        #result = buildref_optimal(imagelist)

    log.info ('wall-time spent in imcombine: {}s'.format(time.time()-t0))


################################################################################

def calc_headers (combine_type, gains, rdnoises, saturates, exptimes, mjds):
    
    gain = np.mean(gains)
    mjd = np.mean(mjds)
    
    if combine_type == 'sum':
        rdnoise = np.sqrt(np.sum(rdnoises**2))
        saturate = np.sum(saturates)
        exptime = np.sum(exptimes)
        
    else:
        rdnoise = np.sqrt(np.sum(rdnoises**2)) / len(rdnoises)
        saturate = np.mean(saturates)
        exptime = np.mean(exptimes)
        
    return gain, rdnoise, saturate, exptime, mjd


################################################################################

def run_remap_local (image_new, image_ref, image_out, image_out_size,
                     gain, log, config=None, resample='Y', resampling_type='LANCZOS3',
                     projection_err=0.001, mask=None, header_only='N',
                     resample_suffix='_resamp.fits', resample_dir='.', dtype='float32',
                     value_edge=0, timing=True, nthreads=0, oversampling=0):
        
    """Function that remaps [image_ref] onto the coordinate grid of
       [image_new] and saves the resulting image in [image_out] with
       size [image_size].
    """
    
    if '/' in image_new:
        # set resample directory to that of the new image
        resample_dir = '/'.join(image_new.split('/')[:-1])
        
    # for testing of alternative way; for the moment switch on but
    # needs some further testing
    run_alt = True
    
    if timing: t = time.time()
    log.info('executing run_remap')

    header_new = read_hdulist (image_new, get_data=False, get_header=True)
    header_ref = read_hdulist (image_ref, get_data=False, get_header=True)
    
    # create .head file with header info from [image_new]
    header_out = header_new[:]
    # copy some keywords from header_ref
    #for key in ['exptime', 'saturate', 'gain', 'rdnoise']:
    #    header_out[key_name] = header_ref[key]

    # delete some others
    for key in ['WCSAXES', 'NAXIS1', 'NAXIS2']:
        if key in header_out: 
            del header_out[key]
    # write to .head file
    with open(image_out.replace('.fits','.head'),'w') as newrefhdr:
        for card in header_out.cards:
            newrefhdr.write(str(card)+'\n')

    size_str = str(image_out_size[1]) + ',' + str(image_out_size[0]) 
    cmd = ['swarp', image_ref, '-c', config, '-IMAGEOUT_NAME', image_out, 
           '-IMAGE_SIZE', size_str, '-GAIN_DEFAULT', str(gain),
           '-RESAMPLE', resample,
           '-RESAMPLING_TYPE', resampling_type,
           '-OVERSAMPLING', str(oversampling),
           '-PROJECTION_ERR', str(projection_err),
           '-NTHREADS', str(nthreads)]

    if run_alt:
        cmd += ['-COMBINE', 'N',
                '-RESAMPLE_DIR', resample_dir,
                '-RESAMPLE_SUFFIX', resample_suffix,
                '-DELETE_TMPFILES', 'N']

    # log cmd executed
    cmd_str = ' '.join(cmd)
    log.info('SWarp command executed:\n{}'.format(cmd_str))

    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (stdoutstr,stderrstr) = process.communicate()
    status = process.returncode
    log.info(stdoutstr)
    log.info(stderrstr)
    if status != 0:
        log.error('SWarp failed with exit code {}'.format(status))
        raise Exception('SWarp failed with exit code {}'.format(status))

    if run_alt:
        image_resample = image_out.replace('_remap.fits', resample_suffix)
        data_resample, header_resample = read_hdulist(image_resample,
                                                      get_header=True)
        # SWarp turns integers (mask images) into floats, so making
        # sure that [data_resample] is in the correct format.  All the
        # inputs are fits image names, so have to include an
        # additional [dtype] input.
        if 'int' in dtype:
            data_resample = (data_resample+0.5).astype(dtype)

        # There should be just a shift between the resampled image and
        # the output image in case of COMBINE='Y', which is just
        # determined by which input pixels will end up in the output
        # image. Determine the "0,0" pixel in the output image that
        # corresponds to "0,0" in the input image:
        ra0, dec0 = WCS(header_resample).all_pix2world(0, 0, 0)
        x0, y0 = WCS(header_out).all_world2pix(ra0, dec0, 0)
        x0, y0 = int(x0+0.5), int(y0+0.5)

        # resampled image is a bit smaller than the original image
        # size
        ysize_resample, xsize_resample = np.shape(data_resample)
        # create zero output image with correct dtype
        data_remap = np.zeros(image_out_size, dtype=dtype)
        # or with value [value_edge] if it is nonzero
        if value_edge != 0:
            data_remap += value_edge        
        # and place resampled image in output image
        data_remap[y0:y0+ysize_resample,
                   x0:x0+xsize_resample] = data_resample

        # write to fits [image_out] with correct header; since name of
        # remapped reference bkg/std/mask image will currently be the
        # same for different new images, this if condition needs to
        # go: 
        #if not os.path.isfile(image_out) or get_par(C.redo,tel):
        header_out['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(image_out, data_remap, header_out, overwrite=True)
    
    if timing:
        log_timing_memory (t0=t, label='run_remap', log=log)

    return

    
################################################################################
            
def buildref_optimal(imagelist):

    start_time1 = os.times()

    # size of image
    ysize = refimage_ysize
    xsize = refimage_xsize

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size,
                                                                       ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    nsubs = centers.shape[0]
    if verbose:
        print ('nsubs', nsubs)

    # number of images
    nimages = len(imagelist)
    
    # prepare arrays;
    # sigma is determined inside nsub loop, so only need 1 dimension
    # of size nimages
    sigma_array = np.ndarray((nimages), dtype='float32')
    # the same goes for fratio in case it is determined across the
    # entire image, but it requires nimages x nsubs shape in case
    # fratio is determined for each subimage.
    fratio_array = np.ndarray((nimages, nsubs), dtype='float32')
    # P and data need 4 dimensions: nimages x nsub x ysize x xsize as
    # P needs to be constructed from the full image initially, then
    # inside the subimage loop the 1st dimension will be summed
    P_array = np.ndarray((nimages, nsubs, ysize_fft, xsize_fft), dtype='float32')
    data_array = np.ndarray((nimages, nsubs, ysize_fft, xsize_fft), dtype='float32')
    
    # these output arrays will be filled with the output summed R and
    # P_R for each subimage
    data_R_full = np.zeros((ysize, xsize), dtype='float32')
    data_PR_full = np.zeros((ysize, xsize), dtype='float32')
    
    # list of keywords to be read from header
    keywords = [key_gain, key_ron, key_satlevel]

    # loop images in imagelist
    for nima, image in enumerate(imagelist):

        if verbose:
            print ('\nimage:', image)
            print ('----------------------------------------------------------------------')

        # base name
        base = image[0:-5]
        # name of resampled image produced by swarp
        image_resampled = image.replace('.fits', resample_suffix)
        
        # read in header of image
        t = time.time()    
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            gain, readnoise, satlevel = read_header_alt (header, keywords)
        if verbose:
            print (keywords)
            print (read_header_alt (header, keywords))
            
        # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
        # ref, psf and background images
        data_array[nima], P_array[nima] = prep_optimal_subtraction_adapted(
            image, header, image_resampled, nsubs)
        
        # get x, y and fratios from matching PSFex stars across entire frame
        # first set basename of first image
        if nima==0: base1 = base
        x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base1+'.psfexcat',
                                                                 base+'.psfexcat',
                                                                 base1+'.sexcat',
                                                                 base+'.sexcat')

        # loop nsubs to fill fratio_array - also needs to be done in
        # case fratio is determined from the entire image
        for nsub in range(nsubs):
            if nima==0:
                fratio_array[nima, nsub] = 1.
            else:
                # take local or full-frame values for fratio
                subcut = cuts_ima[nsub]
                index_sub = ((y_fratio > subcut[0]) & (y_fratio < subcut[1]) & 
                             (x_fratio > subcut[2]) & (x_fratio < subcut[3]))
                if fratio_local and any(index_sub):
                    # get median fratio from PSFex stars across subimage
                    fratio_mean, fratio_std, fratio_median = clipped_stats(fratio[index_sub])
                else:
                    # else for the entire image
                    fratio_mean, fratio_std, fratio_median = clipped_stats(fratio, nsigma=2)
                # fill fratio_array
                fratio_array[nima, nsub] = fratio_median
                if verbose:
                    print ('fratio_mean, fratio_median, fratio_std',
                           fratio_mean, fratio_median, fratio_std)
            

    start_time2 = os.times()

    print ('\nexecuting optimal co-addition ...')
    
    for nsub in range(nsubs):
    
        if timing: tloop = time.time()
        
        if verbose:
            print ('\nNsub:', nsub+1)
            print ('----------')

        # determine clipped mean, median and stddev in
        # this subimage of all images
        for nima in range(nimages):
            mean, stddev, median = clipped_stats(data_array[nima,nsub], nsigma=3)
            if verbose:
                print ('clipped mean, median, stddev', mean, median, stddev)

            # replace low values in subimages
            data_array[nima,nsub][data_array[nima,nsub] <= 0.] = median

            # subtract the median background
            data_array[nima,nsub] -= median

            # replace saturated pixel values with zero
            #data[nsub][data[nsub] > 0.95*satlevel] = 0.

            # fill sigma_array
            sigma_array[nima] = stddev

            
        # call optimal coaddition function
        data_R, data_PR = run_coaddition(data_array[:,nsub], P_array[:,nsub], sigma_array,
                                         fratio_array[:,nsub], nimages)
        
        # put sub images into output frames
        subcut = cuts_ima[nsub]
        fftcut = cuts_fft[nsub]
        y1 = subimage_border
        x1 = subimage_border
        y2 = subimage_border+subimage_size
        x2 = subimage_border+subimage_size
        data_R_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_R[y1:y2,x1:x2]
        data_PR_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_PR[y1:y2,x1:x2]

        if timing: print ('wall-time spent in nsub loop', time.time()-tloop)

    end_time = os.times()
    dt_usr  = end_time[2] - start_time2[2]
    dt_sys  = end_time[3] - start_time2[3]
    dt_wall = end_time[4] - start_time2[4]
    print
    print ("Elapsed user time in {0}:  {1:.3f} sec".format("optcoadd", dt_usr))
    print ("Elapsed CPU time in {0}:  {1:.3f} sec".format("optcoadd", dt_sys))
    print ("Elapsed wall time in {0}:  {1:.3f} sec".format("optcoadd", dt_wall))
    
    dt_usr  = end_time[2] - start_time1[2]
    dt_sys  = end_time[3] - start_time1[3]
    dt_wall = end_time[4] - start_time1[4]
    print
    print ("Elapsed user time in {0}:  {1:.3f} sec".format("total", dt_usr))
    print ("Elapsed CPU time in {0}:  {1:.3f} sec".format("total", dt_sys))
    print ("Elapsed wall time in {0}:  {1:.3f} sec".format("total", dt_wall))
    
    # write full new, ref, D and S images to fits
    fits.writeto('R.fits', data_R_full, clobber=True)
    fits.writeto('PR.fits', data_PR_full, clobber=True)
        
    # and display
    cmd = ['ds9', '-zscale', 'R.fits', 'PR.fits', 'swarp_combined.fits']
    result = subprocess.call(cmd)
    

################################################################################
        
def read_header_alt (header, keywords):
    
    values = []
    for i in range(len(keywords)):
        if keywords[i] in header:
            values.append(header[keywords[i]])
        else:
            raise RuntimeError ('keyword {} not present in header - change keyword '
                                'name or add manually'.format(keywords[i]))
    return values


################################################################################
    
def prep_optimal_subtraction_adapted (image, header, image_resampled, nsubs):

    print ('\nexecuting prep_optimal_subtraction ...')
    t = time.time()
    
    # read in data of resampled image
    with fits.open(image_resampled) as hdulist:
        data = hdulist[0].data
    # get gain, readnoise and pixscale from header
    gain = header[key_gain]
    readnoise = header[key_ron]
    pixscale = header[key_pixscale]
    # convert counts to electrons
    data *= gain

    # determine psf of image (not resampled) with get_psf function
    psf = get_psf_adapted (image, header, nsubs)

    # split full image into subimages
    ysize, xsize = refimage_ysize, refimage_xsize

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        subimage_size, ysize, xsize)

    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    
    fftdata = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    for nsub in range(nsubs):
        subcutfft = cuts_ima_fft[nsub]
        fftcut = cuts_fft[nsub]
        fftdata[nsub][fftcut[0]:fftcut[1],fftcut[2]:fftcut[3]] = data[subcutfft[0]:subcutfft[1],
                                                                      subcutfft[2]:subcutfft[3]]
        
    if timing: print ('wall-time spent in prep_optimal_subtraction', time.time()-t)
    return fftdata, psf
    

################################################################################

def get_psf_adapted (image, ima_header, nsubs):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    """

    if timing: t = time.time()
    print ('\nexecuting get_psf ...')

    # determine image size from header
    xsize, ysize = ima_header['NAXIS1'], ima_header['NAXIS2']
    
    # run psfex on SExtractor output catalog
    sexcat = image.replace('.fits', '.sexcat')
    psfexcat = image.replace('.fits', '.psfexcat')
    if not os.path.isfile(psfexcat) or redo:
        print ('sexcat', sexcat)
        print ('psfexcat', psfexcat)
        result = run_psfex_adapted (sexcat, psfex_cfg, psfexcat)

    # read in PSF output binary table from psfex
    psfex_bintable = image.replace('.fits', '.psf')
    with fits.open(psfex_bintable) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data[0][0][:]

    # read in some header keyword values
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_fwhm = header['PSF_FWHM']
    psf_samp = header['PSF_SAMP']
    # [psf_size_config] is the size of the PSF as defined in the PSFex
    # configuration file ([PSF_SIZE] parameter), which is the same as
    # the size of the [data] array
    psf_size_config = header['PSFAXIS1']
    if verbose:
        print ('polzero1                   ', polzero1)
        print ('polscal1                   ', polscal1)
        print ('polzero2                   ', polzero2)
        print ('polscal2                   ', polscal2)
        print ('order polynomial:          ', poldeg)
        print ('PSF FWHM:                  ', psf_fwhm)
        print ('PSF sampling size (pixels):', psf_samp)
        print ('PSF size defined in config:', psf_size_config)
        
    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border

    # For all images, the PSF was determined from the original image,
    # while it will be applied to the images remapped/resampled to the
    # reference frame. This reference frame is not necessarily that of
    # one of the images. So the centers of the cutouts in the remapped
    # images need to be mapped back to those in the original image to
    # get the PSF from the proper coordinates.  Easiest to do this
    # using astropy.wcs, which would also take care of any potential
    # rotation and scaling.
    
    # first infer ra, dec corresponding to x, y pixel positions
    # (centers[:,1] and centers[:,0], respectively, in the
    # resampled image
    wcs = WCS(image.replace('.fits','_resamp.fits'))
    ra_temp, dec_temp = wcs.all_pix2world(centers[:,1], centers[:,0], 1)
    # then convert ra, dec back to x, y in the original image
    # using the .wcs file produced by Astrometry.net
    wcs = WCS(image.replace('_wcs.fits','.wcs'))
    centers[:,1], centers[:,0] = wcs.all_world2pix(ra_temp, dec_temp, 1)
        
    # [psf_size] is the PSF size in image pixels,
    # i.e. [psf_size_config] multiplied by the PSF sampling (roughly
    # 4-5 pixels per FWHM) which is automatically determined by PSFex
    # (PSF_SAMPLING parameter in PSFex config file set to zero)
    psf_size = np.int(np.ceil(psf_size_config * psf_samp))
    # if this is odd, make it even - for the moment this is because
    # the index range of the bigger image in which this psf is put
    # ([psf_ima_center]) assumes this is even
    if psf_size % 2 != 0:
        psf_size += 1
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)
    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.ndarray((nsubs,psf_size,psf_size), dtype='float32')
    # [psf_ima_center] is [psf_ima] broadcast into images of xsize_fft
    # x ysize_fft
    psf_ima_center = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    # [psf_ima_shift] is [psf_ima_center] shifted - this is
    # the input PSF image needed in the zogy function
    psf_ima_shift = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    
    # loop through nsubs and construct psf at the center of each
    # subimage, using the output from PSFex that was run on the full
    # image
    for nsub in range(nsubs):
        
        x = (centers[nsub,1] - polzero1) / polscal1
        y = (centers[nsub,0] - polzero2) / polscal2

        if nsubs==1:
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + \
                          data[3] * y + data[4] * x * y + data[5] * y**2
            elif poldeg==3:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 + \
                          data[4] * y + data[5] * x * y + data[6] * x**2 * y + \
                          data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3

        if display:
            # write this psf to fits
            fits.writeto('psf_'+imtype+'_sub'+str(nsub)+'.fits', psf_ima_config, clobber=True)
            #result = show_image(psf_ima_config)

        # resample PSF image at image pixel scale
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update)
        psf_ima[nsub] = psf_ima_resized
        if verbose and nsub==1:
            print ('psf_samp, psf_samp_update', psf_samp, psf_samp_update)
            print ('np.shape(psf_ima_config)', np.shape(psf_ima_config))
            print ('np.shape(psf_ima)', np.shape(psf_ima))
            print ('np.shape(psf_ima_resized)', np.shape(psf_ima_resized))
            print ('psf_size ', psf_size)
        if display:
            # write this psf to fits
            fits.writeto('psf_resized_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_resized, clobber=True)
            #result = show_image(psf_ima_resized)

        # normalize to unity
        psf_ima_resized_norm = psf_ima_resized / np.sum(psf_ima_resized)
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            print ('WARNING: image not even in both dimensions!')
            
        xcenter_fft, ycenter_fft = int(xsize_fft/2), int(ysize_fft/2)
        if verbose and nsub==1:
            print ('xcenter_fft, ycenter_fft ', xcenter_fft, ycenter_fft)
            psf_ima_center[nsub,
                           ycenter_fft-int(psf_size/2):ycenter_fft+int(psf_size/2),
                           xcenter_fft-int(psf_size/2):xcenter_fft+int(psf_size/2)] \
                           = psf_ima_resized_norm
            
        if display:
            fits.writeto('psf_center_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_center[nsub], clobber=True)            
            #result = show_image(psf_ima_center[nsub])

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])
        # Eran's function:
        #print np.shape(image_shift_fft(psf_ima_center[nsub], 1., 1.))
        #psf_ima_shift[nsub] = image_shift_fft(psf_ima_center[nsub], 0., 0.)

        #result = show_image(psf_ima_shift[nsub])

    if timing: print ('wall-time spent in get_psf', time.time() - t)

    return psf_ima_shift


################################################################################

def run_psfex_adapted (cat_in, file_config, cat_out):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if timing: t = time.time()

    if psf_sampling == 0:
        # provide new PSF_SIZE based on psf_radius, which is 2 *
        # [psf_radius] * FWHM / sampling factor. The sampling factor is
        # automatically determined in PSFex, and is such that FWHM /
        # sampling factor ~ 4-5, so:
        size = np.int(psf_radius*9+0.5)
        # make sure it's odd
        if size % 2 == 0: size += 1
        psf_size_config = str(size)+','+str(size)
    else:
        # use some reasonable default size
        psf_size_config = '45,45'

    if verbose:
        print ('psf_size_config', psf_size_config)
        
    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config, '-PSF_SAMPLING', str(psf_sampling)]
    result = subprocess.call(cmd)    

    if timing: print ('wall-time spent in run_psfex', time.time()-t)
    

################################################################################

def run_coaddition(M, P, sigma, F, nimages):

    """Function that calculates optimal co-added image (R) and its PSF
       (RP) based on Eqs. 8 and 11 of Zackay & Ofek 2017b, based on
       the inputs (shape):
        - M (nimages, y, x): cube of background-subtracted images
        - P (nimages, y, x): corresponding PSFs of the images
        - sigma (nimages): standard deviation of background
        - F (nimages): flux factors

    """

    if timing: t = time.time()

    numerator = 0.
    denominator2 = 0.
    FR2 = 0.
    
    for j in range(nimages):
        Mj_hat = fft.fft2(M[j])
        Pj_hat = fft.fft2(P[j])
        Pj_hat_conj = np.conj(Pj_hat)
        Pj_hat2_abs = np.abs(Pj_hat**2)
        sigmaj2 = sigma[j]**2
        Fj = F[j]
        Fj2 = F[j]**2

        numerator += (Fj/sigmaj2) * Pj_hat_conj * Mj_hat 
        denominator2 += (Fj2/sigmaj2) * Pj_hat2_abs
        FR2 += (Fj2/sigmaj2)

    denominator = np.sqrt(denominator2)
    R_hat = numerator / denominator
    R = np.real(fft.ifft2(R_hat))

    PR_hat = denominator / np.sqrt(FR2)
    PR = np.real(fft.ifft2(PR_hat))
    
    if timing:
        print ('wall-time spent in optimal coaddition', time.time()-t)
        #print ('peak memory used in run_ZOGY in GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)

    return R, PR


################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='build MeerLICHT/BlackGEM '
                                     'reference images')
    parser.add_argument('--telescope', type=str, default='ML1',
                        choices=['ML1', 'BG2', 'BG3', 'BG4'],
                        help='Telescope name (ML1, BG2, BG3 or BG4); '
                        'default=\'ML1\'')
    parser.add_argument('--date_start', type=str, default=None,
                        help='start date to include images, date string '
                        '(e.g. yyyymmdd) or days relative to now (negative '
                        'number)')
    parser.add_argument('--date_end', type=str, default=None,
                        help='end date to include images, date string '
                        '(e.g. yyyymmdd) or days relative to now (negative '
                        'number)')
    parser.add_argument('--field_ID', type=str, default=None,
                        help='only consider images with this(these) field ID(s) '
                        '(optional use of * and ? wildcards)')
    parser.add_argument('--filters', type=str, default=None,
                        help='only consider this(these) filter(s), e.g. uqi')
    parser.add_argument('--qc_flag_max', type=str, default='yellow',
                        choices=['green', 'yellow', 'orange', 'red'],
                        help='worst QC flag to consider')
    parser.add_argument('--seeing_max', type=float, default=None,
                        help='[arcsec] maximum seeing to consider')
    parser.add_argument('--make_colfig', type=str2bool, default=False,
                        help='make color figures from uqi filters?; '
                        'default=False')
    parser.add_argument('--filters_colfig', type=str, default='iqu',
                        help='set of 3 filters to use for RGB color figures; '
                        'default=\'uqi\'')

    
    
    args = parser.parse_args()

    buildref (telescope = args.telescope,
              date_start = args.date_start,
              date_end = args.date_end,
              field_ID = args.field_ID,
              filters = args.filters,
              qc_flag_max = args.qc_flag_max,
              seeing_max = args.seeing_max,
              make_colfig = args.make_colfig,
              filters_colfig = args.filters_colfig)


################################################################################
