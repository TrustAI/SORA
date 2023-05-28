dataset_dir="/mnt/storage0_8/torch_datasets/mnist-data/"
cuda='0'



model_list='
mnist_relu_6_100
mnist_relu_6_200
mnist_relu_9_200
ffnnRELU__Point_6_500 
convSmallRELU__Point 
convMedGRELU__Point
mnist_conv_maxpool   
convSuperRELU__DiffAI
'

output_dir='/home/fu/workspace/SORA/results/'

width='3'
height='3'
x='13'
y='13'
for model in $model_list; do
    output=${output_dir}${model}"_"${width}"_go.txt"
    line=0
    while [ "$line" -le 100 ]; do
        CUDA_VISIBLE_DEVICES=$cuda python deep_go_mnist_main.py --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --init-k 10 --model-name $model --data-dir $dataset_dir >> $output 
        line=$(( line + 1 ))
    done 
done