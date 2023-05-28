# SORA 

Paper: SORA: Scalable Black-box Reachability Analyser on Neural Networks [[link](https://ieeexplore.ieee.org/abstract/document/10097180)]



All experiments are carried out on a workstation equipped with 96GB RAM, a 20-core Intel i9-10900X CPU, and an Nvidia 2080Ti GPU.

Requirement:
```
PyTorch==1.11.0
torchvision==0.12.0
numpy
```


We used the following pretrained models provided in [ERAN](https://github.com/eth-sri/eran):
```
mnist_relu_6_100
mnist_relu_6_200
mnist_relu_9_200
ffnnRELU__Point_6_500 
convSmallRELU__Point 
convMedGRELU__Point
mnist_conv_maxpool   
convSuperRELU__DiffAI
```
The original models are with the `.onnx` format, so we convert them to fit the PyToch.
The converted version can be download from google drive. [[link](https://drive.google.com/file/d/11HbjUxdUAkzVQ50HtXrmNByyK_fCoPW7/view?usp=sharing)]

Two shell scripts are provided to run the experiments in our paper.
Please check and update the path variables in these scripts before using them.
```bash
mkdir model 
# Download those pretrained models and put them in `model/`
sh run_go.sh
sh run_sora_pgd.sh
```

