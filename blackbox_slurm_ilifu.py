import os

import argparse
#import calendar
import ephem
import glob
import logging
import shutil
import subprocess
import socket
import time

import multiprocessing as mp
import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from datetime import datetime, timedelta
from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

# to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders


# set up log
logfmt = ('%(asctime)s.%(msecs)03d [%(levelname)s, %(process)s] %(message)s '
          '[%(funcName)s, line %(lineno)d]')
datefmt = '%Y-%m-%dT%H:%M:%S'
logging.basicConfig(level='INFO', format=logfmt, datefmt=datefmt)
logFormatter = logging.Formatter(logfmt, datefmt)
logging.Formatter.converter = time.gmtime #convert time in logger to UTC
log = logging.getLogger()
#log.propagate = False
class DropMsgFilter (logging.Filter):
    """drop log records whose message contains any of the given strings"""
    def __init__(self, substrings):
        super().__init__()
        self.substrings = substrings
    def filter(self, record):
        return not any(s in record.getMessage() for s in self.substrings)
drop_noisy = DropMsgFilter(['AdaptiveReadaheadCache enabled'])
for handler in log.handlers:       # the stream handler made by basicConfig
    handler.addFilter(drop_noisy)


__version__ = '0.9.0'


################################################################################
# hardcode the settings below, because they are not accessible on the
# Slurm login node through the usual set_zogy and set_blackbox
# settings files
tel = 'ML1'
data_dir = '/idia/projects/meerlicht'
raw_dir = {}
red_dir = {}
tmp_dir = {}
thumbnails_dir = {}

raw_dir[tel] = '{}/data/raw/{}'.format(data_dir, tel)
red_dir[tel] = '{}/data/red/{}'.format(data_dir, tel)
tmp_dir[tel] = '{}/data/tmp/{}'.format(data_dir, tel)
thumbnails_dir[tel] = '{}/data/thumbnails/{}'.format(data_dir, tel)

ref_dir = '{}/data/ref'.format(data_dir)
log_dir = '{}/RunBlackBOX/log'.format(data_dir)
# folder in which to run/save jobs and logs for the nightly processing
job_dir = '{}/Slurm/{}'.format(log_dir, tel)

# calibration folder and field grid file
cal_dir = '{}/CalFiles'.format(data_dir)
mlbg_fieldIDs = '{}/MLBG_FieldIDs_Feb2022_nGaia.fits'.format(cal_dir)

# MeerLICHT observatory settings
obs_lat = -32.3799
obs_lon = 20.8112
obs_height = 1802
obs_timezone = 'Africa/Johannesburg'

# Email settings
sender = 'BG Admin <bg-admin@blackgem.org>'
# comma-separated email addresses of recipients
recipients = 'ml-nightreports@blackgem.org'
reply_to = 'bg-admin@blackgem.org'
smtp_server = 'smtp-relay.gmail.com'
port = 465
use_SSL = True

save_thumbnails_pngs = True


################################################################################

def adjust_horizon (observer, height):

    # 34arcmin due to atmospheric refraction (ephem uses top of Sun by
    # default, so no need for 16arcmin due to Sun apparent radius)
    horizon = -34/60

    # Earth's radius in m
    R = (1*u.earthRad).to(u.m).value
    horizon -= np.degrees(np.arccos((R/(R+height))))

    # set pressure to zero to discard ephem's internal refraction
    # calculation
    observer.pressure = 0
    observer.horizon = str(horizon)

    return observer


################################################################################

def run_blackbox_slurm (date=None, nthreads=4, mode='night', runtime='6:00:00',
                        force_reproc_new=False, make_masters=True,
                        reduce_science_data=True, update_hdrtables=True,
                        create_nightreports=True):

    # create general logfile based on date/time
    genlogfile = '{}/{}_{}.log'.format(log_dir, tel,
                                       Time.now().strftime('%Y%m%d_%H%M%S'))
    fileHandler = logging.FileHandler(genlogfile, 'a')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel('INFO')
    fileHandler.addFilter(drop_noisy)
    log.addHandler(fileHandler)
    log.info ('general logfile created: {}'.format(genlogfile))


    # if date is not specified, set it to the date of the last local
    # noon; so until noon yesterday's date will be used.
    # SAST is always 2 hours ahead of UT, so can add 2/24 to make the
    # date change at local noon instead of noon UTC.
    date_today = Time(int(Time.now().jd+2/24), format='jd').strftime('%Y%m%d')
    if date is None:
        date = date_today
    else:
        # if it is specified, check if it's today
        date = ''.join(e for e in date if e.isdigit())
        # if not, then run in 'day' mode, i.e. no need to monitor for
        # new files or wait for the end of the night once all images
        # have been processed
        if date != date_today:
            log.info ('input date specified {} is different from today\'s date '
                      '{}; forcing processing to run in day mode'
                      .format(date, date_today))
            mode = 'day'

    global proc_mode
    proc_mode = mode


    # in night mode, all steps are done, in the order: reduce incoming
    # images, make masters, update header tables, create night report
    if mode == 'night':
        make_masters = True
        reduce_science_data = True
        update_hdrtables = True
        create_nightreports = True
    log.info ('mode: {}, make_masters: {}, reduce_science_data: {}, '
              'update_hdrtables: {}, create_nightreports: {}'
              .format(mode, make_masters, reduce_science_data,
                      update_hdrtables, create_nightreports))


    # read field ID table including number of expected Gaia sources
    table_grid = Table.read(mlbg_fieldIDs, memmap=True)

    # create ngaia dictionary with keys int(field_id) and
    # ngaia as values
    ngaia_dict = {}
    for i, field_id in enumerate(table_grid['field_id']):
        ngaia_dict[field_id] = table_grid['ngaia'][i]


    # create list of all fits files already present in [read_path]
    read_path, date_eve = get_path(date, 'read')
    # make sure read_path exists (e.g. if no bias frames were taken,
    # it would not have been created by the transfer script)
    os.makedirs(read_path, exist_ok=True)
    filenames = sorted(glob.glob('{}/{}'.format(read_path, '*fits*')))

    log.info('Slurm-processing in {} mode for evening date: {}'
             .format(mode, date_eve))
    log.info('{} files already present in {}'.format(len(filenames), read_path))

    # prepare folder in which to save and run jobs and logs
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])
    jobnight = '{}/{}'.format(job_dir, date_dir)
    os.makedirs (jobnight, exist_ok=True)


    # set date_begin of reservation
    if mode == 'night':
        date_begin = '{}T{}'.format(date_dir.replace('/','-'), '18:00:00')
    else:
        # in day mode, do not make use of the node reservation, but
        # submit as a general Slurm job
        date_begin = None


    # determine time of next sunrise
    obs = ephem.Observer()
    # MeerLICHT longitude/latitude is hardcoded because there is no
    # access to the zogy settings file and this function is expected
    # to run for the MeerLICHT telescope only
    obs.lat = str(obs_lat)
    obs.lon = str(obs_lon)
    obs.elevation = obs_height
    # correct apparent horizon for observer elevation, which is
    # not taken into account in ephem
    obs = adjust_horizon(obs, obs_height)
    sunrise = obs.next_rising(ephem.Sun())


    # create queue for submitting jobs
    queue = mp.Queue()

    # add files that are already present in the read_path directory to
    # the queue; in night mode all files are added; in day mode only
    # the calibration frames (bias, dark, flat) are added here, and
    # only if the masters are to be made; the science frames are kept
    # aside until the masters have been made
    files_science = []
    for filename in filenames:
        if 'fits' not in filename:
            continue

        if mode == 'night':
            q.put(filename)
        elif is_calib_file(filename):
            if make_masters:
                q.put(filename)
        else:
            files_science.append(filename)


    # create and setup observer, but do not start just yet
    observer = PollingObserver()
    observer.schedule(FileWatcher(queue), read_path, recursive=False)

    # start monitoring [read_path] for incoming files
    observer.start()


    def process_queue():

        """submit a Slurm job for every file in the queue; in night mode
           this keeps monitoring the queue until the night has finished
           and all jobs are done; in day mode it returns once the
           queue is empty. Returns the list of names of the jobs
           submitted.

        """

        # keep monitoring queue - which is being filled with new files
        # detected by watchdog - as long as it is nighttime or the
        # queue is not empty yet
        jobnames_q = []
        #while (ephem.now()-sunrise < 10*ephem.minute or not queue.empty() or
        while (ephem.now()-sunrise < 15*ephem.minute or not queue.empty() or
               # in night mode, also wait until all jobs are finished
               (mode=='night' and len(list_active_jobs(jobnames_q)) > 0)):

            if queue.empty():
                if mode == 'night':
                    time.sleep(10)
                else:
                    # in day mode, no additional files should be coming
                    # in, so can break out of while loop
                    break

            else:
                filename = get_file (queue)
                if filename is not None:

                    # Python command to execute
                    python_cmdstr = (
                        'python /Software/BlackBOX/blackbox.py --img_reduce True '
                        '--cat_extract True --trans_extract True --keep_tmp False '
                        '--force_reproc_new {} --image {}'
                        .format(filename, force_reproc_new))

                    log.info ('python command string to execute: {}'
                              .format(python_cmdstr))


                    # use single thread for calibration files
                    if np.any([s in filename.lower()
                               for s in ['bias', 'flat', 'dark']]):
                        nthreads = 1

                    else:
                        # default number of threads for object images
                        nthreads = 2

                        # if field contains many Gaia sources, use more
                        # threads
                        try:
                            # extract field ID from header
                            header = read_hdulist(filename, get_data=False,
                                                  get_header=True)
                            if 'OBJECT' in header:
                                field_id = int(header['OBJECT'])
                                # set nthreads depending on ngaia
                                ngaia = ngaia_dict[field_id]
                                if ngaia > 2e5:
                                    nthreads = 4

                                if ngaia > 5e5:
                                    nthreads = 6

                                if ngaia > 1e6:
                                    nthreads = 8

                                log.info ('estimated # Gaia sources in field ({}): '
                                          '{}; using {} threads for {}'
                                          .format(field_id, ngaia, nthreads,
                                                  filename))
                        except Exception as e:
                            log.warning ('exception occurred when inferring ngaia '
                                         'for {}; using default nthreads ({}): {}'
                                         .format(filename, nthreads, e))


                    # process it through a SLURM batch job
                    jobname = filename.split('/')[-1].split('.fits')[0]
                    slurm_process (python_cmdstr, nthreads, runtime, jobname,
                                   jobnight, date_begin=date_begin)

                    # append filename to list of files processed
                    jobnames_q.append(jobname)

        return jobnames_q


    # maximum wait time equal to twice the input [runtime]
    wait_max = 2 * np.sum(np.array(runtime.split(':')).astype(int)
                          *np.array([3600, 60, 1]))

    def wait_for (jobnames_wait, label):
        if len(jobnames_wait) == 0:
            return
        log.info ('waiting for {} {} job(s) to finish'
                  .format(len(jobnames_wait), label))
        nsec_wait = wait4jobs2finish (jobnames_wait, wait_max=wait_max)
        log.info ('waited for {:.2f} hours for the {} jobs to finish'
                  .format(nsec_wait/3600, label))


    ############################################################
    # step 1: night mode: reduce all images, including the ones that
    # arrive during the night; day mode: reduce calibration frames
    ############################################################
    jobnames = process_queue()
    
    if mode == 'night':
        log.info ('night has finished and queue is empty')
        # watchdog can be stopped
        observer.stop() #stop observer
        observer.join() #join observer

    wait_for (jobnames, 'image reduction')

    log.info ('night has finished and queue is empty')


    ############################################################
    # step 2: create master frames
    ############################################################
    if make_masters:

        jobnames_masters = []
        # create master frames
        python_cmdstr_master = ('python /Software/BlackBOX/blackbox.py '
                                '--master_date {}'.format(date_eve))
        # process masters through a SLURM batch job; could split this up
        # into several jobs, one for the master bias and one for each of
        # the master flats in a specific filter, but even with a single
        # task this takes only about 10-15 minutes
        jobname_masters = 'masters_{}'.format(date_eve)
        slurm_process (python_cmdstr_master, nthreads=4, runtime='1:00:00',
                       jobname=jobname_masters, jobnight=jobnight,
                       date_begin=date_begin)
        jobnames.append(jobname_masters)
        jobnames_masters.append(jobname_masters[tel])

        # in day mode, science frames are only reduced after the
        # masters have been made
        if mode == 'day' and reduce_science_data:
            wait_for (jobnames_masters, 'master frame')


    ############################################################
    # step 3 (day mode only): reduce the science frames
    ############################################################
    if mode == 'day' and reduce_science_data:

        log.info ('reducing {} science frames'.format(len(files_science)))
        for filename in files_science:
            q.put(filename)

        jobnames_science = process_queue()
        jobnames += jobnames_science

        # header tables and night reports need the reduced science frames
        wait_for (jobnames_science, 'science reduction')
 

    ############################################################
    # step 4: add number of header keywords of newly reduced images to
    # header fits tables
    ############################################################
    if update_hdrtables:

        # add number of header keywords of newly reduced images to big
        # header fits tables
        try:

            # loop different catalogs
            for cat_type in ['cat', 'trans', 'sso']:

                # add night's headers to header file
                fits_header = ('{}/data/hdrtables/{}_headers_{}.fits'
                               .format(data_dir, tel, cat_type))
                path_full = '{}/{}'.format(red_dir, date_dir)
                end_str = ''
                if cat_type == 'cat':
                    search_str = '_red_{}.fits'.format(cat_type)
                else:
                    search_str = '_{}.fits'.format(cat_type)

                python_cmdstr = ('python -c \"import set_blackbox as set_bb; '
                                 'from blackbox import add_headkeys; '
                                 'add_headkeys (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\')\"'
                                 .format(path_full, fits_header,
                                         search_str, end_str, tel))

                jobname = 'add_headkeys_{}_{}'.format(cat_type, date_eve)
                slurm_process (python_cmdstr, nthreads=1, runtime='0:30:00',
                               jobname=jobname, jobnight=jobnight, date_begin=date_begin)
                jobnames.append(jobname)

        except Exception as e:
            log.exception('exception occurred during adding header keys: {}'
                          .format(e))


    ############################################################
    # step 5: create night reports
    ############################################################
    if create_nightreports:

        if mode == 'night':
            screenshot = True
        else:
            # in day mode, do not make screenshot, since it's an older
            # date that is being processed
            screenshot = False

        try:
            if False:
                python_cmdstr = ('python -c \"import set_blackbox as set_bb; '
                                 'from blackbox import create_obslog; '
                                 'create_obslog (\'{}\', email=True, '
                                 'tel=\'{}\', proc_mode=\'{}\', '
                                 ' weather_screenshot={})\"'
                                 .format(date, tel, proc_mode, screenshot))

                jobname = 'obslog_{}'.format(date_eve)
                slurm_process (python_cmdstr, nthreads=1, runtime='0:10:00',
                               jobname=jobname, jobnight=jobnight,
                               date_begin=date_begin)
                jobnames.append(jobname)

            else:
                create_obslog (date, email=True, tel=tel, proc_mode=proc_mode,
                               weather_screenshot=screenshot)

        except Exception as e:
            log.exception('exception occurred while creating {} obslog '
                          'for date {}: {}'.format(tel, date, e))


    # now that night has finished, collect individual logfiles
    # produced by Slurm and append them to the logging for this night
    if False:
        try:
            for jobname in jobnames:
                logfile_slurm = '{}/{}.err'.format(jobnight, jobname)
                if os.path.isfile(logfile_slurm):
                    with open(logfile_slurm, 'r') as f:
                        log.info (f.read())
        except Exception as e:
            log.exception('exception occurred during appending of individual '
                          'logfiles to the general logfile: {}'.format(e))


    log.info('All done')
    
    return


################################################################################

def is_calib_file (filename):

    """True if [filename] is a bias, dark or flat frame"""

    return any(s in filename.lower() for s in ['bias', 'dark', 'flat'])


################################################################################

def wait4jobs2finish (jobnames, wait_max=3600):

    # wait for a bit to make sure any recently submitted jobs have
    # made it to the queue
    t0 = time.time()
    time.sleep(20)

    jobnames_run = jobnames.copy()
    while time.time()-t0 < wait_max:

        # only list jobs that are running or pending
        jobnames_run = list_active_jobs (jobnames_run)
        njobs = len(jobnames_run)

        # if no more running or pending jobs, break
        if njobs==0:
            break
        else:
            log.info ('{} job(s) still running or pending'.format(njobs))

        # wait for a while
        time.sleep(300)

    else:
        log.warning ('maximum wait time of {}s reached'.format(wait_max))


    return time.time()-t0


################################################################################

def list_active_jobs (jobnames, states=['RUNNING', 'PENDING']):

    jobnames_out = []
    for jobname in jobnames:
        state = get_job_state(jobname)
        #log.info ('jobname: {}, state: {}'.format(jobname, state))
        if state in states:
            jobnames_out.append(jobname)

    return jobnames_out


################################################################################

def get_job_state (jobname):

    cmd = ('/opt/slurm/bin/sacct -S now-12hours -E now -o State -n -X --name {} '
           '| tail -1'.format(jobname))
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode('UTF-8').replace('\n','').strip()


################################################################################

def slurm_process (python_cmdstr, nthreads, runtime, jobname, jobnight,
                   date_begin=None, account='b19-meerlicht-ag',
                   reservation='meerlicht'):

    try:

        # create SLURM batch job in date_eve subfolder of nightjobs
        # folder with name based on input jobname
        jobfile = '{}/{}.sh'.format(jobnight, jobname)
        with open(jobfile, 'w') as f:

            f.write ('#!/bin/bash\n')
            f.write ('#SBATCH --nodes=1\n')
            f.write ('#SBATCH --ntasks-per-node=1\n')
            f.write ('#SBATCH --cpus-per-task={}\n'.format(nthreads))
            f.write ('#SBATCH --time={}\n'.format(runtime))
            f.write ('#SBATCH --mem-per-cpu=12GB\n')
            f.write ('#SBATCH --job-name={}\n'.format(jobname))
            f.write ('\n')
            f.write ('#SBATCH --account={}\n'.format(account))

            if False:
                # temporarily using cephfs reservation
                reservation='cephfs'
                f.write ('#SBATCH --reservation={}\n'.format(reservation))

            else:
                if date_begin is not None:
                    f.write ('#SBATCH --begin={}\n'.format(date_begin))
                    f.write ('#SBATCH --reservation={}\n'.format(reservation))
                    #f.write ('#SBATCH --nodelist={}\n'.format(nodelist))
                else:
                    f.write ('#SBATCH --partition=Main\n')

            f.write ('\n')
            f.write ('#SBATCH --open-mode=append\n')
            f.write ('#SBATCH --output={}/{}.out\n'.format(jobnight, jobname))
            f.write ('#SBATCH --error={}/{}.err\n'.format(jobnight, jobname))
            f.write ('#SBATCH --mail-user=danielle@blackgem.org\n')
            f.write ('#SBATCH --mail-type=FAIL,TIME_LIMIT\n')
            f.write ('\n')
            f.write ('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
            f.write ('\n')
            f.write ('echo "date              = $(date)"\n')
            f.write ('echo "hostname          = $(hostname -s)"\n')
            f.write ('echo "working directory = $(pwd)"\n')
            f.write ('\n')
            f.write ('echo "#task allocated   = $SLURM_NTASKS"\n')
            f.write ('echo "#cores/task       = $SLURM_CPUS_PER_TASK"\n')
            f.write ('\n')
            f.write ('/software/common/singularity/4.4.1/bin/singularity exec '
                     '--bind /idia/projects/meerlicht '
                     '--env MLBG_CALDIR={} '
                     '/idia/projects/meerlicht/Containers/MLBG_latest.sif {}\n'
                     .format(cal_dir, python_cmdstr))
            f.write ('\n')



        # make batch script executable
        cmd = ['chmod', '+x', jobfile]
        subprocess.run(cmd)


        # submit batch script
        cmd = ['/opt/slurm/bin/sbatch', jobfile]
        result = subprocess.run(cmd, capture_output=True)
        log.info ('{}, jobfile: {}'.format(result.stdout.decode('UTF-8')
                                           .replace('\n',''), jobfile))
        #log.info ('{}, jobfile: {}'.format(result, jobfile))


        if False:
            # delete batch job
            log.info ('removing jobfile {}'.format(jobfile))
            os.remove(jobfile)


    except Exception as e:

        log.exception ('exception was raised in [slurm_process]: {}'
                       .format(e))


    return


################################################################################

def create_obslog (date, email=True, tel=None, proc_mode=None,
                   weather_screenshot=True):

    # extract table with various observables/keywords from the headers
    # of all raw/reduced files of a particular (evening) date,
    # e.g. ORIGFILE, IMAGETYP, DATE-OBS, PROGNAME, PROGID, OBJECT,
    # FILTER, EXPTIME, RA, DEC, AIRMASS, FOCUSPOS, image quality
    # (PSF-FWHM), QC-FLAG, ..
    #
    # if email==True, also send an email to people that are
    # interested; the email parameters such as sender, recipients,
    # etc., can be defined in BlackBOX settings file


    date_eve = ''.join(e for e in date if e.isdigit())
    if len(date_eve) != 8:
        log.error ('input date to function create_obslog needs to consist of '
                   'at least 8 digits, yyyymmdd, where the year, month and '
                   'day can be connected with any type of character, e.g. '
                   'yyyy/mm/dd or yyyy-mm-dd, etc.')
        return


    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])
    red_path = red_dir[tel]
    full_path = '{}/{}'.format(red_path, date_dir)

    log.info ('full_path: {}'.format(full_path))


    # collect biases, darks, flats and science frames in different
    # lists, using [list_files] as little as possible
    log.info ('collecting lists of reduced bias, dark, flat and science frames')
    all_files_list = list_files('{}'.format(full_path), search_str='.fits',
                                recursive=True)
    bias_list = [fn for fn in all_files_list if '/bias/' in fn]
    dark_list = [fn for fn in all_files_list if '/dark/' in fn]
    flat_list = [fn for fn in all_files_list if '/flat/' in fn]
    object_list = [fn for fn in all_files_list if '_red.fits' in fn]


    filenames = [bias_list, dark_list, flat_list, object_list]
    # clean up [filenames]
    filenames = [f for sublist in filenames for f in sublist]


    # number of different reduced files
    nred = len(filenames)
    nbias_red = len(bias_list)
    ndark_red = len(dark_list)
    nflat_red = len(flat_list)
    nobject_red = len(object_list)

    # number of thumbnail zips
    if save_thumbnails_pngs:
        thumbnails_path = '{}/{}'.format(thumbnails_dir[tel], date_dir)
        zip_list = list_files(thumbnails_path, end_str='.zip')
        nzips = len(zip_list)

        # Older BG data have unzipped directories instead of zips
        if nzips == 0:
            if thumbnails_path[0:5] == "gs://":
                _, path_nobucket = get_bucket_name(thumbnails_path)
            else:
                path_nobucket = thumbnails_path
            path_depth = len([p for p in path_nobucket.split('/') if p])
            if path_depth > 0:
                log.info ('Older data: counted unzipped thumbnail dirs')
            nzips = len(list_folders(thumbnails_path, depth=path_depth+1))

    # collect raw image list
    raw_path = raw_dir[tel]
    log.info ('collecting list of raw frames')
    raw_list = list_files('{}/{}'.format(raw_path, date_dir), search_str='.fits')

    # number of different raw files
    nraw = len(raw_list)
    nbias_raw = 0
    ndark_raw = 0
    nflat_raw = 0
    nobject_raw = 0
    nobject_notracking = 0
    first = True
    for f in raw_list:
        if 'bias' in f.lower():
            nbias_raw += 1
        elif 'dark' in f.lower():
            ndark_raw += 1
        elif 'flat' in f.lower():
            nflat_raw += 1
        elif 'singleobservation' not in f.lower():
            nobject_raw += 1

            if first:
                log.info ('reading headers of raw object frames')
            first = False
            hdr_raw = read_hdulist(f, get_data=False, get_header=True, memmap=True)
            if 'ISTRACKI' in hdr_raw and str2bool(hdr_raw['ISTRACKI']) == False:
                nobject_notracking += 1

    # maximum filename length for column format
    #max_length = max([len(f.strip()) for f in filenames])

    # keywords to add to table
    keys = ['ORIGFILE', 'IMAGETYP', 'DATE-OBS', 'PROGNAME', 'PROGID', 'OBJECT',
            'FILTER', 'EXPTIME', 'RA', 'DEC', 'AIRMASS', 'FOCUSPOS',
            'S-SEEING', 'CL-BASE', 'RH-MAST', 'WINDAVE', 'LIMMAG', 'QC-FLAG',
            'QCRED1', 'QCRED2', 'QCRED3']
    formats = {#'ORIGFILE': '{:60}',
        #'IMAGETYP': '{:<8}',
        'DATE-OBS': '{:.19}',
        'EXPTIME': '{:.1f}',
        'RA': '{:.3f}',
        'DEC': '{:.3f}',
        'AIRMASS': '{:.3f}',
        'S-SEEING': '{:.4}',
        'CL-BASE': '{:.4}',
        'RH-MAST': '{:.4}',
        'WINDAVE': '{:.4}',
        'LIMMAG': '{:.5}'
    }


    # loop input list of filenames
    log.info ('reading headers of reduced frames')
    rows = []
    for filename in filenames:

        fn_hdr = filename.replace('.fits.fz', '_hdr.fits')
        if isfile (fn_hdr):
            file2read = fn_hdr
        else:
            file2read = filename

        # read file header
        header = read_hdulist (file2read, get_data=False, get_header=True)

        # prepare row of filename and header values
        row = []
        for key in keys:
            if key in header:
                row.append(header[key])
            else:
                row.append(' ')

        # append to rows
        rows.append(row)


    # create table from rows
    nobject_noref = 0
    names = []
    for i_key, key in enumerate(keys):
        names.append(key)

    if len(rows) == 0:
        # rows without entries: create empty table
        table = Table(names=names)
    else:
        table = Table(rows=rows, names=names)

        # Count how many images do not have ref images, but ignore the red-
        # flagged images as dummy trans products are created for those
        is_obj = np.array([str(t).strip().lower()=='object'
                           for t in table['IMAGETYP']])
        is_red = np.array([str(q).strip().lower()=='red'
                           for q in table['QC-FLAG']])
        sel = is_obj & ~is_red

        objs = np.array([str(o).strip() for o in table['OBJECT'][sel]])
        filts = np.array([str(f).strip() for f in table['FILTER'][sel]])
        for obj, filt in set(zip(objs, filts)):
            list_ref = list_files('{}/{}'.format(ref_dir, obj),
                                  search_str='_{}_'.format(filt),
                                  end_str='_red.fits.fz')
            if len(list_ref) == 0:
                n_combo = int(np.sum((objs==obj) & (filts==filt)))
                log.info ('No ref for {} in {} ({} images)'
                          .format(obj, filt, n_combo))
                nobject_noref += n_combo

    # order by DATE-OBS
    index_sort = np.argsort(table['DATE-OBS'])
    table = table[index_sort]

    # write table to ASCII file in tmp folder
    tmp_path = tmp_dir[tel]
    # just in case it does not exist yet, create it
    make_dir (tmp_path)
    obslog_tmp = '{}/{}_{}_obslog.txt'.format(tmp_path, tel, date_eve)
    log.info ('saving header info to {}'.format(obslog_tmp))


    # if table is empty, no files were processed and reduced folder
    # needs to be created before writing the empty table
    if len(rows)==0:
        make_dir (full_path)
        ascii.write (table, obslog_tmp, overwrite=True)
    else:
        # write the filled table
        ascii.write (table, obslog_tmp, format='fixed_width_two_line',
                     delimiter_pad=' ', position_char=' ',
                     formats=formats, overwrite=True)


    # for MeerLICHT and BlackGEM, save the weather page as a screen
    # shot in tmp folder, and add it as attachment to the mail
    png_present = False
    if tel=='ML1':
        png_tmp = '{}/{}_SAAOweather.png'.format(tmp_path, date_eve)
        webpage = 'https://suthweather.saao.ac.za'
        width = 1500
        height = 2150

    else:
        png_tmp = '{}/{}_LaSilla_meteo.png'.format(tmp_path, date_eve)

        # Deprecated link:
        # date in format e.g. 28+Oct+2024
        #date_tmp = '{}+{}+{}'.format(date_eve[6:8],
        #                             calendar.month_abbr[int(date[4:6])],
        #                             date[0:4])
        #webpage = ('https://archive.eso.org/asm/ambient-server?'
        #           'night={}&site=lasilla'.format(date_tmp))

        # the following line is only returning a white stripe, perhaps
        # only the very top of the webpage?
        #webpage = ('https://www.eso.org/asm/ui/publicLog?name=LaSilla&startDate='
        #           '{}'.format(date_eve))
        webpage = 'https://www.ls.eso.org/lasilla/dimm/meteomonitor.html'

        width = 1500
        height = 1150

        if proc_mode != 'night':
            try:
                # Get the weather graphs from the archive instead of making a
                # screenshot of last night's weather and instead of using the
                # potentially existing image in the bucket (which, in the past,
                # was the screenshot of the processing night, not necessarily
                # the observing night).
                date_next = (datetime.strptime(date_eve, '%Y%m%d') +
                             timedelta(days=1))
                ps_img = 'end{}.ps'.format(date_next.strftime('%y%m%d'))
                webpage = ('https://www.ls.eso.org/lasilla/dimm/Archive/{}'
                           .format(ps_img))
                cmd = ["wget", "-N", webpage]
                result = subprocess.run(cmd, capture_output=True, timeout=180,
                                        cwd=tmp_path)
                #log.info(result.stderr.decode('UTF-8'))
                with Image.open("{}/{}".format(tmp_path, ps_img)) as img:
                    rotated_img = img.rotate(270, expand=True)
                    rotated_img.save(png_tmp, "PNG")

                # Succesfully retrieved from archive. Won't make a screenshot
                weather_screenshot = False
                remove_files (["{}/{}".format(tmp_path, ps_img)], verbose=True)

            except Exception as e:
                log.exception ('exception occurred while downloading weather '
                               'page {}: {}'.format(webpage, e))

    # define png destination
    png_dest = '{}/{}/{}'.format(red_path, date_dir, png_tmp.split('/')[-1])

    # Don't remake screenshot of last night's weather if screenshot already
    # exists (to prevent overwriting older archival screenshots with the new
    # processing night's screenshot)
    if weather_screenshot and not isfile(png_dest):
        try:

            log.info ('saving screenshot of {} to {}'.format(webpage, png_tmp))
            cmd = ['wkhtmltoimage', '--quiet', '--quality', '80',
                   '--crop-w', str(width), '--crop-h', str(height),
                   webpage, png_tmp]
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            #log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
            #log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))

        except Exception as e:
            log.exception ('exception occurred while making screenshot of '
                           'weather page {}: {}'.format(webpage, e))
            # do not include screenshot in email
            png_tmp = None

    # check if screenshot already exists
    elif isfile(png_dest) and not isfile(png_tmp):
        copy_file (png_dest, png_tmp, move=False)
        png_present = True


    # additional info that could be added to body of email
    # - any raw files that were not reduced?
    # - list any observing gaps, in fractional UT hours
    # - using above gaps, list fraction of night that telescope was observing
    # - list average exposure overhead in seconds

    body  = '{}: summary of {} observations:\n'.format(tel,
                                                       date_dir.replace('/','-'))
    body += '----------------------------------------\n'

    body += ('# raw images:        {} ({} biases, {} darks, {} flats, {} objects)'
             '\n'.format(nraw, nbias_raw, ndark_raw, nflat_raw, nobject_raw))
    body += ('# reduced images:    {} ({} biases, {} darks, {} flats, {} objects)'
             '\n'.format(nred, nbias_red, ndark_red, nflat_red, nobject_red))
    body += ('# untracked objects: {} (no reduced images)\n'
             .format(nobject_notracking))

    # collect full-source, transient and sso catalog lists, using
    # [list_files] as little as possible
    all_cats_list = list_files('{}/{}'.format(full_path, tel), end_str='.fits')

    log.info ('collecting lists of full-source, transient and sso catalogs, and '
              'counting how many of them are flagged red')
    cat_list = [c for c in all_cats_list if c.endswith('_red_cat.fits')]
    body += ('# full-source cats:  {} ({} red-flagged)\n'.format(
        len(cat_list), count_redflags(cat_list)))

    body += ('# no ref image for:  {} objects\n'.format(nobject_noref))

    trans_list = [c for c in all_cats_list if c.endswith('_trans.fits')]
    body += ('# transient cats:    {} ({} red-flagged)\n'.format(
        len(trans_list), count_redflags(trans_list, key='TQC-FLAG')))

    if save_thumbnails_pngs:
        body += ('# thumbnail zips:    {}\n'.format(nzips))

    sso_list = [c for c in all_cats_list if c.endswith('_trans_sso.fits')]
    body += ('# SSO cats:          {} ({} empty)\n'.format(
        len(sso_list), count_redflags(sso_list, key='SDUMCAT')))
    body += '\n'


    # link to weather page
    body += 'Observing conditions: {}\n'.format(webpage)


    if email:
        # email the obslog (with the weather page for MeerLICHT as
        # attachment) to a list of interested people
        try:
            # subject
            subject = '{} night report {}'.format(tel, date_dir.replace('/','-'))

            log.info ('sending email with subject {} to {} using smtp server {} '
                      'on port {}'
                      .format(subject, recipients, smtp_server, port))

            send_email (recipients, subject, body,
                        attachments='{},{}'.format(obslog_tmp, png_tmp),
                        sender=sender, reply_to=reply_to,
                        smtp_server=smtp_server, port=port, use_SSL=use_SSL)

        except Exception as e:
            log.exception('exception occurred during sending of email: {}'
                          .format(e))

    # now that email is sent, move obslog and weather screenshot
    obslog_dest = '{}/{}/{}_{}_obslog.txt'.format(red_path, date_dir,
                                                  tel, date_eve)
    copy_file (obslog_tmp, obslog_dest, move=True)


    if png_tmp and not png_present:
        copy_file (png_tmp, png_dest, move=True)

    return


################################################################################

def count_redflags(catlist, key='QC-FLAG'):

    nredflags = 0

    for catname in catlist:

        fn_hdr = catname.replace('.fits', '_hdr.fits')
        if isfile (fn_hdr):
            file2read = fn_hdr
        else:
            file2read = catname


        # read file header
        header = read_hdulist (file2read, get_data=False, get_header=True)

        # in case of full-source or transient catalog, 'red' should be
        # in the QC-FLAG keyword, but for SSO cat, SDUMCAT being True
        # indicates a red flag; "==True" is required because if the
        # keyword is not a boolean, header[key] will always be True
        if key in header and ('red' in str(header[key]) or header[key]==True):
            nredflags += 1


    return nredflags


################################################################################

def send_email (recipients, subject, body,
                attachments=None,
                sender='Radboud GW Alert <scheduler@blackgem.org>',
                reply_to='bg-admin@blackgem.org',
                smtp_server='smtp-relay.gmail.com',
                port=465, use_SSL=True):

    if use_SSL:
        smtpObj = smtplib.SMTP_SSL(smtp_server, port)
    else:
        smtpObj = smtplib.SMTP(smtp_server, port)

    smtpObj.ehlo()
    send_from = sender
    send_to = recipients.split(',')
    msg = MIMEMultipart()
    msg['from'] = send_from
    msg['to'] = recipients
    msg['reply-to'] = reply_to
    msg['date'] = formatdate(localtime=True)
    msg['subject'] = subject

    if body is None:
        text = ''
    elif isfile(body):
        with open(body, 'r') as f:
            text = f.read()
    else:
        text = body

    msg.attach( MIMEText(text) )

    if attachments is not None:
        att_list = attachments.split(',')
        for attachment in att_list:
            if isfile(attachment):
                part = MIMEBase('application', "octet-stream")
                part.set_payload( open(attachment,"rb").read() )
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment; filename={}'
                                .format(attachment.split('/')[-1]))
                msg.attach(part)

    smtpObj.sendmail(send_from, send_to, msg.as_string())
    smtpObj.close()


################################################################################

def get_path (date, dir_type):

    # define path

    # date can be any of yyyy/mm/dd, yyyy.mm.dd, yyyymmdd,
    # yyyy-mm-dd or yyyy-mm-ddThh:mm:ss.s; if the latter is
    # provided, make sure to set [date_dir] to the date of the
    # evening before UT midnight
    #
    # rewrite this block using astropy.time; very confusing now
    if 'T' in date:
        if '.' in date:
            # rounds date to microseconds as more digits cannot be
            # defined in the format (next line)
            date = str(Time(date, format='isot'))
            date_format = '%Y-%m-%dT%H:%M:%S.%f'
            high_noon = 'T12:00:00.0'
        else:
            date_format = '%Y-%m-%dT%H:%M:%S'
            high_noon = 'T12:00:00'

        date_ut = dt.datetime.strptime(date, date_format).replace(tzinfo=gettz('UTC'))
        date_noon = date.split('T')[0]+high_noon
        date_local_noon = dt.datetime.strptime(date_noon, date_format).replace(
            tzinfo=gettz(obs_timezone))
        if date_ut < date_local_noon:
            # subtract day from date_only
            date = (date_ut - dt.timedelta(1)).strftime('%Y-%m-%d')
        else:
            date = date_ut.strftime('%Y-%m-%d')

    # this [date_eve] in format yyyymmdd is also returned
    date_eve = ''.join(e for e in date if e.isdigit())
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])


    if dir_type == 'read':
        root_dir = raw_dir
    elif dir_type == 'write':
        root_dir = red_dir
    else:
        log.error ('[dir_type] not one of "read" or "write"')

    path = '{}/{}'.format(root_dir, date_dir)
    if '//' in path:
        log.info ('replacing double slash in path name: {}'.format(path))
        path = path.replace('//','/')

    return path, date_eve


################################################################################

def get_file (queue):

    """Get file from queue after making sure it arrived completely; None
    is returned if the file is not a fits file or still having trouble
    reading the fits file even after waiting for 60s; otherwise the
    filename is returned.

    """

    # get event from queue
    event = queue.get(True)

    try:
        # get name of new file
        filename = str(event.src_path)
        filetype = 'new'

    except AttributeError as e:
        # instead of event, queue entry is a filename added in
        # [run_blackbox]
        filename = event
        filetype = 'pre-existing'


    log.info ('detected a {} file: {}'.format(filetype, filename))


    # only continue if a fits file
    if 'fits' not in filename:

        log.info ('{} is not a fits file; skipping it'.format(filename))
        filename = None

    else:

        # if filename is a temporary rsync copy (default
        # behaviour of rsync is to create a temporary file
        # starting with .[filename].[randomstr]; can be
        # changed with option "--inplace"), then let filename
        # refer to the eventual file created by rsync
        fn_head, fn_tail = os.path.split(filename)
        if fn_tail[0] == '.':
            filename = '{}/{}'.format(fn_head, '.'
                                      .join(fn_tail.split('.')[1:-1]))
            log.info ('changed filename from rsync temporary file {} to {}'
                      .format(event.src_path, filename))

        # this while loop below replaces the old [copying]
        # function; it times out after wait_max is reached
        wait_max = 180
        t0 = time.time()
        time.sleep(1)
        nsleep = 0
        while time.time()-t0 < wait_max:

            try:
                # read the file
                data = read_hdulist(filename, memmap=None)

            except:

                process = False
                if nsleep==0:
                    log.warning ('file {} has not completely arrived yet; '
                                 'will keep trying to read it in for {}s'
                                 .format(filename, wait_max))

                # give file a bit of time to arrive before next read attempt
                time.sleep(5)
                nsleep += 1

            else:
                # if fits file was read fine, set process flag to True
                process = True
                log.info ('successfully read file {} within {:.1f}s'
                          .format(filename, time.time()-t0))
                # and break out of while loop
                break


        if not process:
            log.info ('{}s limit for reading file reached, not processing {}'
                      .format(wait_max, filename))
            filename = None


    return filename


################################################################################

class FileWatcher(FileSystemEventHandler, object):
    '''Monitors directory for new files.

    :param queue: multiprocessing queue for new files
    :type queue: multiprocessing.Queue'''

    def __init__(self, queue):
        self._queue = queue

    def on_created(self, event):
        '''Action to take for new files.

        :param event: new event found
        :type event: event'''
        self._queue.put(event)


################################################################################

def read_hdulist (fits_file, get_data=True, get_header=False,
                  ext_name_indices=None, dtype=None, columns=None,
                  memmap=True):

    """Function to read the data (if [get_data] is True) and/or header (if
    [get_header] is True) of the input [fits_file].  The fits file can
    be an image or binary table, and can be compressed (with the
    compressions that astropy.io can handle, such as .gz and .fz
    files). If [ext_name_indices] is defined, which can be an integer,
    a string matching the extension's keyword EXTNAME or a list or
    numpy array of integers, those extensions are retrieved.

    """

    if os.path.exists(fits_file):
        fits_file_read = fits_file

    else:
        # if fits_file does not exist, look for compressed versions or
        # files without the .fz or .gz extension
        if os.path.exists('{}.fz'.format(fits_file)):
            fits_file_read = '{}.fz'.format(fits_file)
        elif os.path.exists(fits_file.replace('.fz','')):
            fits_file_read = fits_file.replace('.fz','')
        elif os.path.exists('{}.gz'.format(fits_file)):
            fits_file_read = '{}.gz'.format(fits_file)
        elif os.path.exists(fits_file.replace('.gz','')):
            fits_file_read = fits_file.replace('.gz','')
        else:
            raise FileNotFoundError ('file not found: {}'.format(fits_file))


    # open fits file into hdulist
    with fits.open(fits_file_read, memmap=memmap) as hdulist:

        n_exts = len(hdulist)

        # if [ext_name_indices] is a range, or list or numpy ndarray
        # of integers, loop over these extensions and concatenate the
        # data into one astropy Table; it is assumed the extension
        # formats are identical to one another - this is used to read
        # specific extensions from e.g. the calibration catalog.
        if type(ext_name_indices) in [list, range, np.ndarray]:

            for i_ext, ext in enumerate(ext_name_indices):

                # get header from first extension as they should be
                # all identical, except for NAXIS2 (nrows)
                if get_header and i_ext==0:
                    header = hdulist[ext].header

                if get_data:
                    # read extension
                    data_temp = hdulist[ext].data
                    # convert to table, as otherwise concatenation of
                    # extensions below using [stack_arrays] is slow
                    data_temp = Table(data_temp)
                    # could also read fits extension into Table directly,
                    # but this is about twice as slow as the 2 steps above
                    #data_temp = Table.read(fits_file_read, hdu=ext)
                    if i_ext==0:
                        data = data_temp
                    else:
                        #data = stack_arrays((data, data_temp),asrecarray=True,
                        #                    usemask=False)
                        # following does not work if data is a fitsrec
                        # array and the array contains boolean fields, as
                        # these are incorrectly converted; therefore the
                        # conversion to a Table above
                        data = np.concatenate([data, data_temp])

        else:
            # otherwise read the extension defined by [ext_name_indices]
            # or simply the last extension
            if type(ext_name_indices) in [int, str]:
                ext = ext_name_indices
            else:
                ext = n_exts-1

            if get_data:
                data = hdulist[ext].data
                # convert to [dtype] if it is defined
                if dtype is not None:
                    data = data.astype(dtype, copy=False)

            if get_header:
                header = hdulist[ext].header


    if columns is not None:
        # only return defined columns
        return [data[col] for col in columns if col in data.dtype.names]
    else:
        # return data and header depending on whether [get_data]
        # and [get_header] are defined or not
        if get_data:
            if get_header:
                return data, header
            else:
                return data
        else:
            if get_header:
                return header
            else:
                return


################################################################################

def get_bucket_name (path):

    """infer bucket- and filename from [path], which is expected
       to be gs://[bucket name]/some/path/file or [bucket
       name]/some/path/file; if [path] starts with a forward slash,
       empty strings will be returned"""

    bucket_name = path.split('gs://')[-1].split('/')[0]
    if len(bucket_name) > 0:
        # N.B.: returning filename without the starting '/'
        bucket_file = path.split(bucket_name)[-1][1:]
    else:
        bucket_file = ''

    return bucket_name, bucket_file


################################################################################

def isfile (filename):

    if filename[0:5] == 'gs://':

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (filename)
        # N.B.: bucket_file should not start with '/'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(bucket_file)
        return blob.exists()

    else:

        return os.path.isfile(filename)


################################################################################

def isdir (folder):

    if folder[0:5] == 'gs://':

        if folder[-1] != '/':
            folder = '{}/'.format(folder)

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (folder)
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=None, max_results=1)
        nblobs = len(list(blobs))

        return nblobs > 0

    else:

        return os.path.isdir(folder)


################################################################################

def list_folders (path, search_str='', depth=None):

    """function to return list of existing folders starting with
       [path] (can be a folder or google cloud bucket name and path;
       this does not have to be a precise or complete folder/path,
       e.g. [path] can be some_path/some_file_basename), optionally
       containing [search_str] without any wildcards. [depth]
       determines how many folder depths are returned, starting from
       the root directory, where only the folders at that precise
       depth are returned and not their parent folders; for google
       cloud buckets the root folder (depth=1) is considered to start
       after the bucket name, i.e. gs://[bucket name]/[root
       folder]/etc. If [depth] is not defined, all existing folders
       are returned.

    """

    # use function [list_files] to list all files containing 'search_str'
    file_list = list_files (path, search_str=search_str, recursive=True)


    # running in google cloud?
    google_cloud = (path[0:5]=='gs://')


    # remove potential 'gs://' and bucket name
    if google_cloud:

        # infer bucket name
        bucket_name, __ = get_bucket_name(path)

        # chop off 'gs://[bucket_name]'
        file_list = [fn.split(bucket_name)[-1] for fn in file_list]


    if depth is None:
        # list all existing folders
        folder_list = ['/'.join(fn.split('/')[:-1]) for fn in file_list]
    else:
        # create folder list with depth [depth]
        folder_list = ['/'.join(fn.split('/')[:depth+1]) for fn in file_list
                       # line added by Danielle to make sure only
                       # folders with this precise depth are included
                       # also for the linux file system
                       if len(fn.rstrip('/').split('/')) > depth]


    # remove duplicate folders and sort
    folder_list = sorted(list(set(folder_list)))


    # append 'gs://[bucket_name]' if needed
    if google_cloud:
        folder_list = ['gs://{}{}'.format(bucket_name, fn) for fn in folder_list]


    # only folders, not files; added by Danielle as she noticed on the
    # normal (linux) file system also files were returned
    isfolder = [isdir(fn) for fn in folder_list]
    folder_list = np.array(folder_list)[isfolder].tolist()


    #log.info ('folder_list: {}'.format(folder_list))


    return folder_list


################################################################################

def list_files (path, search_str='', end_str='', start_str=None,
                recursive=False):

    """function to return list of files starting with [path] (can be a
       folder or google cloud bucket name and path; this does not have
       to be a precise or complete folder/path, e.g. [path] can be
       some_path/some_file_basename), with possible [end_str] and
       containing [search_str] without any wildcards. If path is an
       exact folder, then [start_str] can be used as the beginning of
       the filename.

    """

    # is google cloud being used?
    google_cloud = (path[0:5] == 'gs://')


    # split input [path] into folder_bucket and prefix; if path is a
    # folder, need to add a slash at the end, otherwise the prefix
    # will be the name of the deepest folder
    if isdir(path) and path[-1] != '/':
        path = '{}/'.format(path)


    # if path is indeed a folder, then prefix will be an empty string,
    # which is fine below
    folder_bucket, prefix = os.path.split(path.split('gs://')[-1])


    # if path consists of just the bucket name including gs:// or just
    # a path without any slashes at all, the above will lead to an
    # empty folder_bucket and prefix=path; turn these around
    if len(folder_bucket)==0:
        folder_bucket = prefix
        prefix = ''


    # if prefix is empty and [start_str] is defined, use
    # that as the prefix
    if prefix=='' and start_str is not None:
        prefix = start_str


    # if not dealing with google cloud buckets, use glob
    if not google_cloud:

        #glob files
        if recursive:
            files = glob.glob('{}/**/{}*{}*{}'.format(folder_bucket, prefix,
                                                      search_str, end_str),
                              recursive=True)
            if path in files:
                files.remove(path)

        else:
            files = glob.glob('{}/{}*{}*{}'.format(folder_bucket, prefix,
                                                   search_str, end_str))

    else:

        # for buckets, use storage.Client().list_blobs; see
        # https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix#storage_list_files_with_prefix-python

        # setting delimiter to '/' restricts the results to only the
        # files in a given folder
        if recursive:
            delimiter = None
        else:
            delimiter = '/'


        # bucket name and prefix (e.g. gs://) to add to output files
        bucket_name, bucket_file = get_bucket_name(path)
        bucket_prefix = path.split(bucket_name)[0]


        if False:
            log.info ('folder_bucket: {}'.format(folder_bucket))
            log.info ('prefix: {}'.format(prefix))
            log.info ('path: {}'.format(path))
            log.info ('bucket_name: {}'.format(bucket_name))
            log.info ('bucket_file: {}'.format(bucket_file))


        # get the blobs
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=delimiter)

        # collect blobs' names in list of files
        files = []
        for blob in blobs:

            filename = blob.name

            # check for search string; if search string is empty,
            # following if statement will be False
            if search_str not in filename:
                continue

            # check if filename ends with [end_str]
            len_ext = len(end_str)
            if len_ext > 0 and filename[-len_ext:] != end_str:
                # if not, continue with next blob
                continue

            # after surviving above checks, append filename including
            # the bucket prefix and name
            files.append('{}{}/{}'.format(bucket_prefix, bucket_name, filename))


    #log.info ('files returned by [list_files]: {}'.format(files))
    #log.info ('number of files returned by [list_files]: {}'.format(len(files)))


    return files


################################################################################

def copy_file (src_file, dest, move=False, verbose=True):

    """function to copy or move a file [src_file] to [dest], which may
       be a file or folder; [src_file] and/or [dest] may be part of
       the usual filesystem or in a google cloud bucket; in the latter
       case the argument(s) should start with gs://[bucket_name]

    """

    if verbose:
        if move:
            label = 'moving'
        else:
            label = 'copying'

        log.info('{} {} to {}'.format(label, src_file, dest))


    # if not dealing with google cloud buckets, use shutil.copy2 or shutil.move
    if not (src_file[0:5] == 'gs://' or dest[0:5] == 'gs://'):

        if not move:
            shutil.copy2(src_file, dest)
        else:
            shutil.move(src_file, dest)

    else:

        # this could be done in python, but much easier with gsutil
        # from the shell
        if move:
            cp_cmd = 'mv'
        else:
            cp_cmd = 'cp'


        cmd = ['gsutil', '-q', cp_cmd, src_file, dest]
        result = subprocess.run(cmd)
        #result = subprocess.run(cmd, capture_output=True)
        #log.info(result.stdout.decode('UTF-8'))


    return


################################################################################

def make_dir (path, empty=False):

    """Function to make directory. If [empty] is True and the
       directory already exists, it will first be removed. In case of
       google cloud version, don't do anything - there are no
       actual directories in a bucket."""


    # check if google_cloud is set
    if not path[0:5] == 'gs://':

        # if already exists but needs to be empty, remove it first
        if isdir(path) and empty:
            shutil.rmtree(path)

        # do not check if directory exists, just make it; changed this
        # after racing condition occurred on the ilifu Slurm cluster
        # when reducing flatfields, where different tasks need to make
        # the same directory
        os.makedirs(path, exist_ok=True)


    return


################################################################################

def remove_files (filelist, verbose=False):


    # make sure files exist
    filelist = [f for f in filelist if isfile(f)]
    if len(filelist)==0:
        #log.warning ('no existing file(s) to remove')
        return


    # assume all files are either google cloud or not, so sufficient
    # to only check the 1st file
    google_cloud = (filelist[0][0:5]=='gs://')


    if verbose:
        for f in filelist:
            log.info ('removing file {}'.format(f))


    if not google_cloud:

        hostname = socket.gethostname()
        for f in filelist:
            if 'coma' in hostname and 'astro11' in f:
                cmd = ['ssh', 'astro11-srv', 'rm', f]
                result = subprocess.run(cmd)
            else:
                os.remove(f)

    else:
        cmd = ['gcloud', 'storage', 'rm', '--read-paths-from-stdin',
               '--no-user-output-enabled']
        result = subprocess.run(cmd, input='\n'.join(filelist).encode('utf-8'))


    return


################################################################################

# from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################################################################

def main ():

    """Wrapper allowing [run_blackbox_slurm] to be run from the command line"""

    parser = argparse.ArgumentParser(description='Run BlackBOX on ilifu Slurm '
                                     'cluster')
    parser.add_argument('--date', type=str, default=None,
                        help='date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd '
                        'or yyyy.mm.dd); default=today (date change is at local noon)')
    parser.add_argument('--nthreads', type=int, default=4,
                        help='number of threads/CPUs to use per image; default=4')
    parser.add_argument('--mode', choices=['day', 'night'], default='night',
                        help='processing mode; night mode will also process new '
                        'incoming images; default=night')
    parser.add_argument('--runtime', type=str, default='6:00:00',
                        help='runtime requested per image; default=6:00:00')
    parser.add_argument('--force_reproc_new', type=str2bool, default=False,
                        help='force reprocessing (overwriting) of image; '
                        'default=False')

    # Day-mode switches (in night mode, all steps are executed)
    parser.add_argument('--make_masters', type=str2bool, default=True,
                        help='reduce biases/flats and make masters before ; '
                        'reducing science data. Only relevant in day mode; '
                        'default=True. In night mode, biases/flats are reduced'
                        'alongside science images and masters are only made '
                        'after processing all images.')
    parser.add_argument('--reduce_science_data', type=str2bool, default=True,
                        help='reduce science images; default=True. Always True'
                        ' in night mode.')
    parser.add_argument('--update_hdrtables', type=str2bool, default=True,
                        help='Update header tables; default=True. Always True '
                        'in night mode.')
    parser.add_argument('--create_nightreports', type=str2bool, default=True,
                        help='Create night reports; default=True. Always True '
                        'in night mode.')
    args = parser.parse_args()


    run_blackbox_slurm (args.date, args.nthreads, args.mode, args.runtime,
                        args.force_reproc_new, make_masters=args.make_masters,
                        reduce_science_data=args.reduce_science_data,
                        update_hdrtables=args.update_hdrtables,
                        create_nightreports=args.create_nightreports)


################################################################################

if __name__ == "__main__":
    main()
