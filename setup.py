from setuptools import setup, find_packages
setup(
    name='blackbox',
    version='0.7.7',
    description='image processing sofware specifically written for the reduction of BlackGEM and MeerLICHT images',
    url='https://github.com/pmvreeswijk/BlackBOX',
    author='Paul Vreeswijk, Kerry Paterson',
    author_email='pmvreeswijk@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'matplotlib', 'scipy', 'pyfftw',
                      'lmfit', 'sip_tpv', 'scikit-image',
                      # dependencies above: zogy
                      # dependencies below: blackbox
                      'astroscrappy', 'acstools', 'ephem', 'watchdog']
)
