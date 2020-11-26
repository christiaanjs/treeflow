#!/bin/bash
set -eo pipefail

SCRIPT=`realpath $0`
TREEFLOW_DIR=`dirname $SCRIPT`

cd $TREEFLOW_LIB

# Install Beagle hmc-clock
git clone -b hmc-clock --single-branch https://github.com/beagle-dev/beagle-lib.git
cd beagle-lib
./autogen.sh
./configure --prefix=$TREEFLOW_LIB
make install

# Set up libsbn environment
cd $TREEFLOW_LIB
git clone -b 264-ratio-gradient-jacobian --single-branch https://github.com/phylovi/libsbn.git
cd libsbn
conda env create -f environment.yml
conda activate libsbn
git submodule update --init --recursive

# Add Beagle to libsbn environment
BEAGLE_PREFIX=$TREEFLOW_LIB
ACTIVATE_DIR=$CONDA_PREFIX/etc/conda/activate.d
DEACTIVATE_DIR=$CONDA_PREFIX/etc/conda/deactivate.d
mkdir -p $ACTIVATE_DIR $DEACTIVATE_DIR
echo "export BEAGLE_PREFIX=$BEAGLE_PREFIX" >> $ACTIVATE_DIR/vars.sh
echo 'export LD_LIBRARY_PATH=$BEAGLE_PREFIX/lib:$LD_LIBRARY_PATH' >> $ACTIVATE_DIR/vars.sh
echo 'export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | perl -pe "s[$BEAGLE_PREFIX][]g;")' >> $DEACTIVATE_DIR/vars.sh
echo "unset BEAGLE_PREFIX" >> $DEACTIVATE_DIR/vars.sh
# Install libsbn
conda activate libsbn
scons
make

cd $TREEFLOW_DIR
pip install -e .