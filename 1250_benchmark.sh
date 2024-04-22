#!/bin/bash

# ResNet configurations
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 10 --res_layers 18 --log_file ./logs/1k/ResNet18_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 1 --res_layers 18 --log_file ./logs/1k/ResNet18_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer adam --lr 0.0025 --res_layers 18 --log_file ./logs/1k/ResNet18_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 10 --res_layers 18 --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 1 --res_layers 18 --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer adam --lr 0.0025 --res_layers 18 --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 10 --res_layers 18 --reduction_layers True --log_file ./logs/1k/ResNet18_R_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 1 --res_layers 18 --reduction_layers True --log_file ./logs/1k/ResNet18_R_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer adam --lr 0.0025 --res_layers 18 --reduction_layers True --log_file ./logs/1k/ResNet18_R_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 10 --res_layers 18 --reduction_layers True --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_R_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer sgd --lr 1 --res_layers 18 --reduction_layers True --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_R_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model resnet --optimizer adam --lr 0.0025 --res_layers 18 --reduction_layers True --pretrained True --log_file ./logs/1k/ResNet18_Pretrained_R_Adam_LR0025.log

# QResNet configurations
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --repetitions 6 --log_file ./logs/1k/QResNet18_C6_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --repetitions 6 --log_file ./logs/1k/QResNet18_C6_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --repetitions 6 --log_file ./logs/1k/QResNet18_C6_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --repetitions 6 --pretrained True --log_file ./logs/1k/QResNet18_C6_Pretrained_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --repetitions 6 --pretrained True --log_file ./logs/1k/QResNet18_C6_Pretrained_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --repetitions 6 --pretrained True --log_file ./logs/1k/QResNet18_C6_Pretrained_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --repetitions 4 --log_file ./logs/1k/QResNet18_C4_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --repetitions 4 --log_file ./logs/1k/QResNet18_C4_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --repetitions 4 --log_file ./logs/1k/QResNet18_C4_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --pretrained True --log_file ./logs/1k/QResNet18_Pretrained_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --pretrained True --log_file ./logs/1k/QResNet18_Pretrained_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --pretrained True --log_file ./logs/1k/QResNet18_Pretrained_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --repetitions 2 --log_file ./logs/1k/QResNet18_C2_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --repetitions 2 --log_file ./logs/1k/QResNet18_C2_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --repetitions 2 --log_file ./logs/1k/QResNet18_C2_Adam_LR0025.log

python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 10 --res_layers 18 --repetitions 2 --pretrained True --log_file ./logs/1k/QResNet18_C2_Pretrained_SGD_LR10.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer sgd --lr 1 --res_layers 18 --repetitions 2 --pretrained True --log_file ./logs/1k/QResNet18_C2_Pretrained_SGD_LR1.log
python main.py --batch_size 64 --n_dataset_examples 1250 --epochs 20 --model qresnet --optimizer adam --lr 0.0025 --res_layers 18 --repetitions 2 --pretrained True --log_file ./logs/1k/QResNet18_C2_Pretrained_Adam_LR0025.lo