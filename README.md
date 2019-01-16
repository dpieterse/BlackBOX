# BlackBOX
Image processing sofware specifically written for the reduction of BlackGEM and MeerLICHT images. It is based on Kerry Paterson's **BGreduce**. BlackBOX performs standard CCD image reduction tasks on multiple images simultaneously using multi-processing and multi-threading, and feeds the reduced images to **ZOGY** to ultimately perform optimal image subtraction and detect transient sources.

It makes grateful use of the following programs that first need to be installed:

- acstools.satdet: https://acstools.readthedocs.io/en/stable/satdet.html
- astroscrappy: https://github.com/astropy/astroscrappy

This project is licensed under the terms of the MIT license.
