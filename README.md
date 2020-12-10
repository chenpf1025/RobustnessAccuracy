# Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels.
This is the official repository for the paper [Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels](https://arxiv.org/abs/2012.04193).
```
@article{chen2020robustness,
  title={Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels.},
  author={Chen, Pengfei and Ye, Junjie and Chen, Guangyong and Zhao, Jingwei and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2012.04193},
  year={2020}
}
```

Under diagonally-dominant class-conditional label noise, the main conclusions are as follows.
* A classifier maximizing its accuracy on the noisy distribution is guaranteed to maximize the accuracy on clean distribution.
* We can obtain an optimal classifier by maximizing training accuracy on sufficiently many noisy samples.
* A noisy validation set is reliable.

Regarding the the critical demand of model selection in scenarios like hyperparameter-tuning and early stopping, we theoretically prove that a noisy validation set is reliable. We empritically verify the utility of *a noisy validation set* by showing the impressive performance of a very simple method *Noisy best Teacher and Student (NTS)*.
<div align=center><img src="https://github.com/chenpf1025/robustness_accuracy/blob/master/table.png" width = "100%"/></div>


## Requirements
* Python 3.6+
* PyTorch 1.2+
* torchvision 0.4+
* pillow 5.0+
* numpy 1.17+

## Noisy best Teacher and Student (NTS)
| File                          | Usage                                                      |
|-------------------------------|------------------------------------------------------------|
|hyperparameter.txt             | containing detailed hyperparameters                        |
| command.txt                   | containing commands for running the training               |
| train_cifar10_ce.py           | training on CIFAR-10 with normal cross-entropy (NT and NS) |
| train_cifar10_ct.py           | training on CIFAR-10 with Co-teaching (NT and NS)          |
| train_cifar10_dmi.py          | training on CIFAR-10 with DMI (NT and NS)                  |
| train_cifar10_gce.py          | training on CIFAR-10 with GCE (NT and NS)                  |
| train_cifar100_ce.py          | training on CIFAR-10 with normal cross-entropy (NT and NS) |
| train_cifar100_ct.py          | training on CIFAR-100 with Co-teaching (NT and NS)         |
| train_cifar100_dmi.py         | training on CIFAR-100 with DMI (NT and NS)                 |
| train_cifar100_gce.py         | training on CIFAR-100 with GCE (NT and NS)                 |
| train_clothing1m_ce.py        | training on Clothing1M with normal cross-entropy (NS)      |
| train_clothing1m_dividemix.py | training on Clothing1M with DivideMix (NT)                 |



## Example commands
Please refer to **hyperparameter.txt** and **command.txt** for detailed hyperparameters and commands.

#### CIFAR: training teacher and get NT
```
**CE CIFAR-10 Teacher**
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.2 --save_model
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.4 --save_model
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.6 --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.2 --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.3 --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.4 --save_model
```

```
**CE CIFAR-100 Teacher**
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.2 --save_model
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.4 --save_model
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.6 --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.2 --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.3 --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.4 --save_model
```

```
**GCE CIFAR-10 Teacher**
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.2 --save_model
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.4 --save_model
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.6 --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.2 --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.3 --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.4 --save_model
```

```
**GCE CIFAR-100 Teacher**
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.2 --save_model
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.4 --save_model
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.6 --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.2 --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.3 --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.4 --save_model
```

```
**Co-T CIFAR-10 Teacher**
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.2 --tau 0.2 --save_model
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.4 --tau 0.4 --save_model
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.6 --tau 0.6 --e_warm 60 --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.2 --tau 0.1 --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.3 --tau 0.15 --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.4 --tau 0.2 --e_warm 60 --save_model
```

```
**Co-T CIFAR-100 Teacher**
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.2 --tau 0.2 --save_model
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.4 --tau 0.4 --save_model
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.6 --tau 0.6 --e_warm 60 --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.2 --tau 0.2 --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.3 --tau 0.3 --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.4 --tau 0.4 --e_warm 60 --save_model
```

```
**DMI CIFAR-10 teacher** (DMI requires a pretrained model, e.g., using CE, to initialize)
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.2 --init_path cifar10_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.4 --init_path cifar10_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.6 --init_path cifar10_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.2 --init_path cifar10_asym0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.3 --init_path cifar10_asym0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.4 --init_path cifar10_asym0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**DMI CIFAR-100 Teacher**
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.2 --init_path cifar100_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.4 --init_path cifar100_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.6 --init_path cifar100_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.2 --init_path cifar100_pair0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.3 --init_path cifar100_pair0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.4 --init_path cifar100_pair0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

#### CIFAR: training student and get NS
```
**CE CIFAR-10 Student**
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.2 --teacher_path cifar10_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.4 --teacher_path cifar10_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ce.py --noise_pattern uniform --noise_rate 0.6 --teacher_path cifar10_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.2 --teacher_path cifar10_asym0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.3 --teacher_path cifar10_asym0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ce.py --noise_pattern asym --noise_rate 0.4 --teacher_path cifar10_asym0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**CE CIFAR-100 Student**
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.2 --teacher_path cifar100_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.4 --teacher_path cifar100_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ce.py --noise_pattern uniform --noise_rate 0.6 --teacher_path cifar100_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.2 --teacher_path cifar100_pair0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.3 --teacher_path cifar100_pair0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ce.py --noise_pattern pair --noise_rate 0.4 --teacher_path cifar100_pair0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**GCE CIFAR-10 Student**
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.2 --teacher_path gce_cifar10_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.4 --teacher_path gce_cifar10_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_gce.py --noise_pattern uniform --noise_rate 0.6 --teacher_path gce_cifar10_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.2 --teacher_path gce_cifar10_asym0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.3 --teacher_path gce_cifar10_asym0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_gce.py --noise_pattern asym --noise_rate 0.4 --teacher_path gce_cifar10_asym0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**GCE CIFAR-100 Student**
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.2 --teacher_path gce_cifar100_uniform0.2_dp0.0_augstandard_seed0_best.pth --save_model
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.4 --teacher_path gce_cifar100_uniform0.4_dp0.0_augstandard_seed0_best.pth --save_model
python train_cifar100_gce.py --noise_pattern uniform --noise_rate 0.6 --teacher_path gce_cifar100_uniform0.6_dp0.0_augstandard_seed0_best.pth --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.2 --teacher_path gce_cifar100_pair0.2_dp0.0_augstandard_seed0_best.pth --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.3 --teacher_path gce_cifar100_pair0.3_dp0.0_augstandard_seed0_best.pth --save_model
python train_cifar100_gce.py --noise_pattern pair --noise_rate 0.4 --teacher_path gce_cifar100_pair0.4_dp0.0_augstandard_seed0_best.pth --save_model
```

```
**Co-T CIFAR-10 Student**
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.2 --tau 0.05 --teacher_path ct_cifar10_uniform0.2_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.4 --tau 0.1 --teacher_path ct_cifar10_uniform0.4_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ct.py --noise_pattern uniform --noise_rate 0.6 --tau 0.15 --e_warm 60 --teacher_path ct_cifar10_uniform0.6_warm60_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.2 --tau 0.03 --teacher_path ct_cifar10_asym0.2_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.3 --tau 0.04 --teacher_path ct_cifar10_asym0.3_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_ct.py --noise_pattern asym --noise_rate 0.4 --tau 0.05 --e_warm 60 --teacher_path ct_cifar10_asym0.4_warm60_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**Co-T CIFAR-100 Student**
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.2 --tau 0.1 --teacher_path ct_cifar100_uniform0.2_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.4 --tau 0.2 --teacher_path ct_cifar100_uniform0.4_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ct.py --noise_pattern uniform --noise_rate 0.6 --tau 0.3 --e_warm 60 --teacher_path ct_cifar100_uniform0.6_warm60_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.2 --tau 0.1 --teacher_path ct_cifar100_pair0.2_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.3 --tau 0.2 --teacher_path ct_cifar100_pair0.3_warm0_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_ct.py --noise_pattern pair --noise_rate 0.4 --tau 0.3 --e_warm 60 --teacher_path ct_cifar100_pair0.4_warm60_dp0.2_augstrong_seed0_best.pth --save_model      
```

```
**DMI CIFAR-10 Student**
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.2 --init_path cifar10_uniform0.2_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.4 --init_path cifar10_uniform0.4_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern uniform --noise_rate 0.6 --init_path cifar10_uniform0.6_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.2 --init_path cifar10_asym0.2_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_asym0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.3 --init_path cifar10_asym0.3_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_asym0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar10_dmi.py --noise_pattern asym --noise_rate 0.4 --init_path cifar10_asym0.4_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar10_asym0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

```
**DMI CIFAR-100 Student**
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.2 --init_path cifar100_uniform0.2_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_uniform0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.4 --init_path cifar100_uniform0.4_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_uniform0.4_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern uniform --noise_rate 0.6 --init_path cifar100_uniform0.6_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_uniform0.6_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.2 --init_path cifar100_pair0.2_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_pair0.2_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.3 --init_path cifar100_pair0.3_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_pair0.3_dp0.2_augstrong_seed0_best.pth --save_model
python train_cifar100_dmi.py --noise_pattern pair --noise_rate 0.4 --init_path cifar100_pair0.4_dp0.2_augstrong_seed0_best.pth --teacher_path dmi_cifar100_pair0.4_dp0.2_augstrong_seed0_best.pth --save_model
```

#### Clothing1M: training teacher and student
```
**Teacher and Student: Clothing1M**
python train_clothing1m_dividemix.py --root data/Clothing1M/
python train_clothing1m_ce.py --root data/Clothing1M/ --teacher_path dividemix_net1.pth
python train_clothing1m_ce.py --root data/Clothing1M/ --teacher_path dividemix_net2.pth
```

```
**Teacher and Student: Clothing1M (Noisy Validation)**
python train_clothing1m_dividemix.py --root data/Clothing1M/ --use_noisy_val
python train_clothing1m_ce.py --root data/Clothing1M/ --teacher_path nv_dividemix_net1.pth --use_noisy_val
python train_clothing1m_ce.py --root data/Clothing1M/ --teacher_path nv_dividemix_net2.pth --use_noisy_val
```
