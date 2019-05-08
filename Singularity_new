BootStrap: debootstrap

OSVersion: stable

MirrorURL: http://ftp.us.debian.org/debian/

%labels


  package.name

  package.version 0.1.0

  package.homepage

  package.source.url

  package.source.mdm5

  package.license GPLv3

%post

    apt-get update

    apt-get -y install python2.7 python-pip

    apt-get -y install python-tk

    apt-get -y install git

    apt-get -y install libgsl-dev
   
    apt-get install python-dev

    pip install numpy

    python -m pip install -U matplotlib

    pip install emcee

    pip install corner

    pip install pandas

    pip install seaborn

    pip install cython

    pip install pybind11

    pip install autograd

    git init

    git clone https://github.com/dfm/celerite.git

    cd /celerite

    python setup.py install

    pip install george

    pip install pyfits

    pip install pywcs

    cd /usr/local

    git init

    git clone https://github.com/as595/pyrmsynth.git

    cd pyrmsynth/rm_tools/

    python setup.py build_ext --inplace

    apt-get clean

%runscript

    cd /mnt

    python /usr/local/pyrmsynth/rmsynthesis.py "$@"
