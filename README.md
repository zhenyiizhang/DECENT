# Genetic deconvolution of embryonic and maternal cell-free DNA in spent medium of human preimplantation embryos through deep learning


[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/zhenyiizhang/DECENT/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/zhenyiizhang/DECENT?color=blue)](https://github.com/zhenyiizhang/DECENT/)


## Introduction
We present DECENT (**de**ep **C**NV r**e**co**n**s**t**ruction), a new deep learning framework aimed at mitigating maternal contamination in spent culture embryo media (SECM) and reconstructing embryonic copy number variations (CNVs). DECENT leverages sequence and methylation information from both embryonic and maternal sources, utilizing convolutional neural networks and attention mechanisms to infer the origin of sequence reads. Overall, DECENT contributes to substantially enhancing the diagnostic accuracy and effectiveness of SECM-based niPGT, establishing a robust groundwork for the extensive clinical utilization of niPGT in the field of reproductive medicine.

<p align="center">
  <img src=https://github.com/zhenyiizhang/DECENT/blob/main/figures/main.svg alt="[main]">
</p>

## Get Started

1. Clone this repo:
```bash
git clone https://github.com/zhenyiizhang/DECENT
cd DECENT
```

- You can create a new conda  environment (DECENT) using 
```bash
conda env create -f environment.yml
```

- In MAC OS, there's an issue with installing pysam-related packages. Below is an alternative way to install in the conda environment:
```bash
conda env create -f environment_mac.yml
conda activate DECENT
pip install pysam
```

2. Both are also needed to install bedtools related packages, it's recommended to refer to [this source](https://bedtools.readthedocs.io/en/latest/content/installation.html) for more details.

3. If you want to conduct CNV analysis further, you should config [Ginkgo](https://github.com/robertaboukhalil/ginkgo) on your own server and then use our reference samples and scripts replacing the original for analysis. The new scripts can be found under the directory ```Ginkgo-scripts```, please use it to relpace ```scripts``` in original [Ginkgo](https://github.com/robertaboukhalil/ginkgo) directory. Note that the root need to be accordingly changed to your own directory. Furtherwore, we provide on some intructions on the cofigurations of Ginkgo:
- /etc/php.ini: not need to do anything.
- ginkgo/includes/fileupload/server/php/UploadHandler.php:  ```upload_dir='/lustre/home/2301110060/ginkgo-master/uploads/'```, ```upload_url='http://gb.cshl.edu/ginkgo/uploads/'```
- ginkgo/bootstrap.php: ```DIR_ROOT='/lustre/home/2301110060/ginkgo-master'```, ```URL_ROOT='http://gb.cshl.edu/ginkgo'```
- The rest of root directory cofigurations can refer to the code in ```Ginkgo-scripts```.

## How to use

If the model has been trained, we then demonstrate how to utilize the model to eliminate contamination and perform copy number variations analysis. (Code in ```infer``` directory)

1. In ```config.py```, please change ```data_dir``` to your own data directory. The ```reads_dir```, ```result_dir```, ```processed_bam_dir``` are the path to store the results. The data directory structure looks like this:
```
│
├── data              <- data dir
|   ├── PBAT_Sxxx_Bxx
        ├── PBAT_Sxxx_Bxx.rmdup.bam                   
│   ├── PBAT_Sxxx_Bxx     
        ├── PBAT_Sxxx_Bxx.rmdup.bam    
```
2. Then you can run the code to process one sample by 
```
bash job_cnv.sh PBAT_Sxxx_Bxx
```
Then ```result/processed_bam``` will store the processed bam and bed files after contamination removal.

3. We then can use the processed bed files to conduct the CNV analysis. You should create a uploads directory for example ```upload_dir='/lustre/home/2301110060/ginkgo-master/uploads/'```. Then you can create a directory under it like ```test```, then you should move the bed files you want to do on this directory. Then you should create a ```list``` file, a ```config``` file and a reference bam file therein. We provide  examples on ```Ginkgo-uploads``` directory. The reference bam file can be found in . Then you can proceed CNV analysis by
```
bash analyze.sh DIR/uploads/test
```
## How to train


## Contact information

- Zhenyi Zhang (SMS, PKU)-[zhenyizhang@stu.pku.edu.cn](mailto:zhenyizhang@stu.pku.edu.cn)
- Peijie Zhou (CMLR, PKU) (Corresponding author)-[pjzhou@pku.edu.cn](mailto:pjzhou@pku.edu.cn)
- Yidong Chen (Thid Hospitial, PKU) (Corresponding author)-[chenyidongahu@163.com](mailto:chenyidongahu@163.com)

## License
DECENT is licensed under the MIT License. 

```
MIT License

Copyright (c) 2024 Zhenyi Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


