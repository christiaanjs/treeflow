#!/bin/bash
set -euo pipefail

SCRIPT=`realpath $0`
TREEFLOW_DIR=`dirname $SCRIPT`

cd $TREEFLOW_LIB

# Install Beagle hmc-clock
git clone -b hmc-clock --single-branch https://github.com/beagle-dev/beagle-lib.git
cd beagle-lib
./autogen.sh
./configure --prefix=$TREEFLOW_LIB
make install

# Install libsbn
cd $TREEFLOW_LIB
git clone -b 264-ratio-gradient-jacobian --single-branch https://github.com/phylovi/libsbn.git
cd libsbn
conda env create -f environment.yml
conda activate libsbn
git submodule update --init --recursive
export BEAGLE_PREFIX=$TREEFLOW_LIB
scons
conda activate libsbn
make

cd $TREEFLOW_DIR
pip install -e .