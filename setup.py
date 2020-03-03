from setuptools import setup, find_packages
setup(
    name='blackbox',
    version='0.9.2',
    description='image processing sofware specifically written for the reduction of BlackGEM and MeerLICHT images',
    url='https://github.com/pmvreeswijk/BlackBOX',
    author='Paul Vreeswijk, Kerry Paterson, Danielle Pieterse',
    author_email='pmvreeswijk@gmail.com',
    python_requires='>=3',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'matplotlib', 'scipy', 'pyfftw',
                      'lmfit', 'sip_tpv', 'scikit-image', 'bottleneck',
                      # dependencies above: zogy
                      # dependencies below: blackbox
                      'astroscrappy', 'acstools', 'ephem', 'watchdog',
                      'fitsio', 'memory-profiler', 'aplpy']
)
