dataset_dir="/mnt/storage0_8/torch_datasets/mnist-data/"
cuda='1'


# convSmallRELU__PGDK
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

    direct
    nb_iter='150'
    nb_eval='20000'
    depth='4'

    output=${output_dir}${model}"_"${width}"_direct.txt"
    line=0
    while [ "$line" -le 99 ]; do
        CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --obstacle --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir --cw >> $output 
        line=$(( line + 1 ))
    done 

    #pgd
    pgd_iter='100'
    restart='5'
    step_size='0.01'

    output=${output_dir}${model}"_"${width}"_pgd.txt"
    line=0
    while [ "$line" -le 99 ]; do
        CUDA_VISIBLE_DEVICES=$cuda python first_order_mnist_main.py --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $pgd_iter --step-size $step_size --restart $restart --model-name $model --data-dir $dataset_dir >> $output 
        line=$(( line + 1 ))
    done 
done

# width='6'
# height='6'
# x='11'
# y='11'
# for model in $model_list; do

#     # direct
#     nb_iter='150'
#     nb_eval='10000'
#     depth='5'

#     output=${output_dir}${model}"_"${width}"_direct.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --obstacle --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 

#     #pgd
#     pgd_iter='150'
#     restart='5'
#     step_size='0.01'

#     output=${output_dir}${model}"_"${width}"_pgd.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python first_order_mnist_main.py --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $pgd_iter --step-size $step_size --restart $restart --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 
# done

# width='12'
# height='12'
# x='8'
# y='8'
# for model in $model_list; do

#     # direct
#     nb_iter='150'
#     nb_eval='10000'
#     depth='4'

#     output=${output_dir}${model}"_"${width}"_direct.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --obstacle --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 

#     #pgd
#     pgd_iter='100'
#     restart='5'
#     step_size='0.01'

#     output=${output_dir}${model}"_"${width}"_pgd.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python first_order_mnist_main.py --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $pgd_iter --step-size $step_size --restart $restart --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 
# done

# width='28'
# height='28'
# x='0'
# y='0'
# for model in $model_list; do

#     # direct
#     nb_iter='150'
#     nb_eval='20000'
#     depth='3'

#     output=${output_dir}${model}"_"${width}"_direct.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --obstacle --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 

#     #pgd
#     pgd_iter='100'
#     restart='5'
#     step_size='0.01'

#     output=${output_dir}${model}"_"${width}"_pgd.txt"
#     line=0
#     while [ "$line" -le 100 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python first_order_mnist_main.py --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $pgd_iter --step-size $step_size --restart $restart --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 
# done

# width='28'
# height='28'
# x='0'
# y='0'
# for model in $model_list; do

#     # direct
#     nb_iter='200'
#     nb_eval='60000'
#     depth='20'

#     output=${output_dir}${model}"_"${width}"_direct_60000_20.txt"
#     line=0
#     while [ "$line" -le 99 ]; do
#         CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --obstacle --example-idx $line --topleft-x $x --topleft-y $y --width $width --height $height --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir >> $output 
#         line=$(( line + 1 ))
#     done 
# done