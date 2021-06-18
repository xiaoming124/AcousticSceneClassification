#!/usr/bin/env bash

net="asd_baseline"
CPU_ARCH="arm64"
DNNC_MODE="debug"
dnndk_board="ZCU102"
dnndk_dcf="dpu-02-09-2020-9-30.dcf"

echo "Compiling Network ${net}"

# Work space directory
work_dir=$(pwd)

# Path of caffe quantization model
# model_dir=${work_dir}/quantize_results
# Output directory
output_dir="dnnc_output"

tf_model="./quantize_results/deploy_model.pb"

DNNC=dnnc
echo "CPU Arch   : $CPU_ARCH"
echo "DNNC Mode  : $DNNC_MODE"
echo "$(dnnc --version)"
$DNNC   --parser=tensorflow                         \
       --frozen_pb=${tf_model}                     \
       --output_dir=${output_dir}                  \
       --dcf=${dnndk_dcf}                          \
       --mode=${DNNC_MODE}                         \
       --cpu_arch=${CPU_ARCH}                      \
       --net_name=${net}

