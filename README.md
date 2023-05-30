## Federated Affective Computing with Label Distribution Skew 
* **Pytorch implementation for paper:** Knowledge-guided Label Distribution Calibration for Federated Affective Computing


## Usage
### 0. Installation

- Run `cd FACLDS/`
- Install the libraries listed in requirements.txt 


### 1. Prepare Dataset 

We use ResNet3D to extract video features.

The extracted features can be divided into data using ```/utils/noniid_split.py ```


### 2. Train Model

```
python main.py --FL_platform MLP-FACLDS --net_name MLP --dataset youtube --E_epoch 1 --max_communication_rounds 300 --split_type c_5_a_5 --save_model_flag
```

- All the checkpoints, results, log files will be saved to the ```--output_dir``` folder, with the final performance saved at log_file.txt 



## Acknowledgments
- The implementation of our code is based on [Vision Transformer in Federated Learning](https://github.com/Liangqiong/ViT-FL-main)







