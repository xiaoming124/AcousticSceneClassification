### 
# @Author: xiaobo
 # @Email: 729527658@qq.com
 # @Date: 2020-04-20 
 # @Description: quantize frozon tf model 
 # @Dependence: tensorflow 1.13, Vitis-AI Release 1.1
 ###
#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
#conda activate decent_q3

# generate calibraion images and list file
#python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
decent_q  quantize \
  --input_frozen_graph ./frozon_result/model.pb \
  --input_nodes conv2d_1_input \
  --input_shapes ?,40,500,1 \
  --output_nodes dense_2/Softmax \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 50 \
  --output_dir ./quantize_results \

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

