#!/bin/bash

# Bash script to help install BlackBOX and ZOGY automatically on a
# Linux machine. It has been tested using a Google cloud VM instance
# with a fresh install of Ubuntu 18.04 LTS.
#
# to run: download and execute "./startup_blackbox_zogy.sh"
#
# Still to do:
#
# - add the calibration binary fits catalog used by zogy
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading


# versions/settings
# ================================================================================

# python version
v_python="3.7"
# zogy and blackbox; for latest version, leave these empty ("") or comment out
v_blackbox="0.8"
v_zogy="0.8"

# define home of zogy, data and blackbox
zogyhome=${PWD}/ZOGY
datahome=${PWD}
blackboxhome=${PWD}/BlackBOX


# check Linux version, update/upgrade it and set package manager
# ================================================================================

uname_str=$(uname -a)
echo ${uname_str}
if [[ ${uname_str,,} == *"ubuntu"* ]]
then
    packman="apt-get"
    # update
    sudo ${packman} -y update
#elif [[ ${uname_str,,} == *"fedora"* ]]
#then
#    packman="dnf"
fi
# upgrade
sudo ${packman} -y upgrade


# python, pip and git
# ================================================================================

sudo ${packman} -y install python${v_python} python${v_python}-dev
if [ ${v_python} \< "3" ]
then
    sudo ${packman} -y install python-pip
else
    sudo ${packman} -y install python3-pip
fi
pip="python${v_python} -m pip"

# git
sudo ${packman} -y install git


# clone ZOGY and BlackBOX repositories in current directory
# ================================================================================

if [ ! -z ${v_zogy} ]
then
    zogy_branch="--branch v${v_zogy}"
    v_zogy_git="@v${v_zogy}"
fi
git clone ${zogy_branch} https://github.com/pmvreeswijk/ZOGY

if [ ! -z ${v_blackbox} ]
then
    blackbox_branch="--branch v${v_blackbox}"
    v_blackbox_git="@v${v_blackbox}"
fi
git clone ${blackbox_branch} https://github.com/pmvreeswijk/BlackBOX

# rsync BlackBOX content to ZOGY
rsync -av ${blackboxhome}/* ${zogyhome}
# remove BlackBOX to avoid confusion
rm -rf ${blackboxhome}


# install ZOGY and BlackBOX repositories
# ================================================================================

sudo -H ${pip} install git+git://github.com/pmvreeswijk/ZOGY${v_zogy_git}
sudo -H ${pip} install git+git://github.com/pmvreeswijk/BlackBOX${v_blackbox_git}


# download calibration catalog
# ================================================================================

# From where? To be put in ${ZOGYHOME}/CalFiles/


# packages used by ZOGY
# ================================================================================

# Astrometry.net
sudo ${packman} -y install astrometry.net

# SExtractor (although it seems already installed automatically)
sudo ${packman} -y install sextractor
# the executable for this installation is 'sextractor' while ZOGY
# expects 'sex'; make a symbolic link
sudo ln -s /usr/bin/sextractor /usr/bin/sex

# SWarp
sudo ${packman} -y install swarp
# the executable for this installation is 'SWarp' while ZOGY expects
# 'swarp'; make a symbolic link
sudo ln -s /usr/bin/SWarp /usr/bin/swarp

# PSFEx - this basic install does not allow multi-threading
sudo ${packman} -y install psfex

# ds9
sudo ${packman} -y install saods9

# dfits
sudo ${packman} -y install qfits-tools


# set environent variables:
# -------------------------
if [[ ${SHELL} == *"bash"* ]]
then
    export ZOGYHOME=${zogyhome}
    export DATAHOME=${datahome}
    if [ -z "${PYTHONPATH}" ]
    then
	export PYTHONPATH=${zogyhome}\:${zogyhome}/Settings
    else
	export PYTHONPATH=${PYTHONPATH}\:${zogyhome}\:${zogyhome}/Settings
    fi
elif [[ ${SHELL} == *"csh"* ]]
then
    setenv ZOGYHOME ${zogyhome}
    setenv DATAHOME ${datahome}
    if [ -z "$PYTHONPATH" ]
    then
	setenv PYTHONPATH ${zogyhome}\:${zogyhome}/Settings
    else
	setenv PYTHONPATH ${PYTHONPATH}\:${zogyhome}\:${zogyhome}/Settings
    fi
fi
echo
echo "======================================================================"
echo 
echo "copy and paste the commands below to your shell startup script"
echo "(~/.bashrc, ~/.cshrc or ~/.zshrc) for these system variables"
echo "and python alias to be set at next login, e.g. for bash:"
echo
echo "# BlackBOX and ZOGY system variables"
echo "export ZOGYHOME=${zogyhome}"
echo "export DATAHOME=${datahome}"
echo "if [ -z \"\${PYTHONPATH}\" ]"
echo "then"
echo "    export PYTHONPATH=${zogyhome}:${zogyhome}/Settings"
echo "else"
echo "    export PYTHONPATH=\${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings"
echo "fi"
echo
echo "# alias for python:"
echo "alias python=python${v_python}"
echo
echo "======================================================================"
echo


# use Astrometry.net link to DR2 files
url="http://data.astrometry.net/5000"

# make sure index files are saved in the right directory; on mlcontrol
# these are in /usr/local/astrometry/data/ (config file:
# /usr/local/astrometry/etc/astrometry.cfg) while on GCloud VM
# installation they are supposed to be in /usr/share/astrometry
# (config file: /etc/astrometry.cfg)
dir1="/usr/share/astrometry"
dir2="/usr/local/astrometry/data"
dir3="${HOME}/IndexFiles"
if [ -d "${dir1}" ]
then
    dir_save=${dir1}
elif [ -d "${dir2}" ]
then
    dir_save=${dir2}
else
    dir_save=${dir3}
    mkdir ${dir3}
fi
echo "downloading Astrometry.net index files to directory ${dir_save}"
echo 
sudo wget -nc $url/index-500{5..6}-0{0..9}.fits -P ${dir_save}
sudo wget -nc $url/index-500{5..6}-1{0..1}.fits -P ${dir_save}

