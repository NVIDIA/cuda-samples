GPU=$1
DRIVER_VERSION=$2
BINARY=./cudaGraphsPerfScaling
datadir=PERF_DATA

suffix=$DRIVER_VERSION
prefix=$GPU
mkdir -p $datadir

trials=600

width=1
nvidia-smi > $datadir/${prefix}_info_${suffix}.txt
$BINARY 5 $trials 1 $width 0 1 256 > $datadir/${prefix}_${width}_small_${suffix}.csv
$BINARY 5 $trials 1 $width 0 32 2048 > $datadir/${prefix}_${width}_large_${suffix}.csv
width=4
$BINARY 5 $trials 1 $width 0 1 256 > $datadir/${prefix}_${width}_small_${suffix}.csv
