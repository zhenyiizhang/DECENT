<h3 align="center"> Genetic deconvolution of embryonic and maternal cell-free DNA in spent medium of human preimplantation embryos through deep learning (Advanced Science)


[Paper Link](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202412660)

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-License-green.svg?labelColor=gray)](https://github.com/zhenyiizhang/DECENT/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/zhenyiizhang/DECENT?color=blue)](https://github.com/zhenyiizhang/DECENT/)
</div>

## Introduction

We present DECENT (**de**ep **C**NV r**e**co**n**s**t**ruction), a deep learning method to reconstruct embryonic copy number variations (CNVs) and mitigate maternal contamination in spent embryo culture media (SECM) of human preimplantation embryos from single-cell methylation sequencing of cell-free DNA (cfDNA). DECENT integrates sequence features and methylation patterns to infer the origin of cfDNA reads. The benchmarking study demonstrated DECENT's ability to estimate contamination proportions and restore embryonic chromosome aneuploidies in samples with varying contamination levels. In highly contaminated SECM clinical samples with more than 80% maternal reads, DECENT achieved consistent embryonic chromosome aneuploidy recovery with invasive tests. Overall, DECENT contributes to enhancing the diagnostic accuracy and effectiveness of noninvasive cfDNA-based preimplantation genetic testing, establishing a robust groundwork for its extensive clinical utilization in the field of reproductive medicine.

<p align="center">
  <img src=https://github.com/zhenyiizhang/DECENT/blob/main/figures/overview.png alt="[main]">
</p>

## Get Started

1. Clone this repository:
```bash
git clone https://github.com/zhenyiizhang/DECENT
cd DECENT
```

- You can create a new conda  environment (DECENT) using 
```bash
conda create -n DECENT python=3.10
pip install -r requirements.txt
```
- For MAC OS, below is the way to install in the conda environment:
```bash
conda env create -f environment.yaml
conda activate DECENT
pip install pysam
```

1. Both are also needed to install bedtools related packages, it's recommended to refer to [this source](https://bedtools.readthedocs.io/en/latest/content/installation.html) for more details.

2. If you want to conduct CNV analysis further, you should config [Ginkgo](https://github.com/robertaboukhalil/ginkgo) on your own server and then use our reference samples and scripts replacing the original for analysis. The new scripts can be found under the directory ```Ginkgo-scripts```, please use them to relpace files in ```scripts``` directory of original [Ginkgo](https://github.com/robertaboukhalil/ginkgo) software. Note that the root need to be accordingly changed to your own directory. We next provide some intructions on the cofigurations of Ginkgo:
- /etc/php.ini: not need to do anything.
- ginkgo/includes/fileupload/server/php/UploadHandler.php:  ```upload_dir='/lustre/home/2301110060/ginkgo-master/uploads/'```, ```upload_url='http://gb.cshl.edu/ginkgo/uploads/'```
- ginkgo/bootstrap.php: ```DIR_ROOT='/lustre/home/2301110060/ginkgo-master'```, ```URL_ROOT='http://gb.cshl.edu/ginkgo'```
- The rest of root directory cofigurations can refer to the code in ```Ginkgo-scripts```.

## How to use

If the model has been trained, we then demonstrate how to utilize the model to eliminate contamination and perform copy number variations analysis. (Code in ```infer``` directory)

1. In ```config.py```, please change ```data_dir``` to your own data directory. The ```reads_dir```, ```score_dir```, ```processed_bam_dir``` are also needed to change into the path to store the results. The data directory structure looks like this:
```
|
|- test
|  |- data  <- data dir
|  |  |- PBAT_Sxxx_Bxx 
|  |  |  |- PBAT_Sxxx_Bxx.rmdup.bam
|  |  |- PBAT_Sxxx_Bxx 
|  |     |- PBAT_Sxxx_Bxx.rmdup.bam
|  |- result
|     |- processed_bam  <- processed_bam_dir
|     |- reads          <- reads_dir
|     |- score          <- score_dir
|  
```

1. Then you can run the code to process one sample by 
```
bash job_cnv.sh PBAT_Sxxx_Bxx
```
Then ```result/processed_bam``` will store the processed bam and bed files after contamination removal.

1. We then can use the processed bed files to conduct the CNV analysis. You should create a uploads directory for example ```upload_dir='/lustre/home/2301110060/ginkgo-master/uploads/'```. Then you can create a directory under it like ```CNV```, then you should move the bed files you want to do to this directory. Then you should create a ```list``` file, a ```config``` file and a reference bed file therein. We provide  examples on ```Ginkgo-uploads``` directory. The reference bed file can be found in https://drive.google.com/drive/folders/1Rdbluc3mqtJHHqMNmKgKu9iPUwzWddFS?usp=sharing (```merge_ref_015_process1.bed``` used for the processed SECM samples, ```merge_20ICM_4M.bed``` used for the original SECM samples). Then you can proceed CNV analysis by
```
bash analyze.sh CNV
```
## How to train
To use your own data to train the model, please find the code in ```train```. You only need to create your own training reads, you can use the code in ```infer```, e.g., ```1_id_bam.py``` and ```2_extract_header.py``` to help you prepare the needed data form. Then you can train your model via
```
python training.py
```
We recommend an Nvidia GPU with CUDA support for training.

## Data availability 
The data that support the findings of this study are openly available in Na- tional Genomics Data Center of the China National Center for Bioinfor- mation at https://ngdc.cncb.ac.cn/gsa-human/browse/HRA000332. Addtionally,
to faciliate access, we also provide a link to download the processed bam files directly at https://drive.google.com/drive/folders/1xmlCocSfwzhH7rZjsXqND3M4on0P0UTe?usp=share_link.

## Contact information

- Zhenyi Zhang (SMS, PKU)-[zhenyizhang@stu.pku.edu.cn](mailto:zhenyizhang@stu.pku.edu.cn)
- Peijie Zhou (CMLR, PKU) (Corresponding author)-[pjzhou@pku.edu.cn](mailto:pjzhou@pku.edu.cn)
- Yidong Chen (Thid Hospital, PKU) (Corresponding author)-[chenyidongahu@163.com](mailto:chenyidongahu@163.com)

## How to cite

If you find DECENT useful in your research, please consider citing our work at this [link](https://advanced.onlinelibrary.wiley.com/action/showCitFormats?doi=10.1002%2Fadvs.202412660).

## License
DECENT is licensed under the following License. 

```
License

Copyright (c) 2025 Zhenyi Zhang

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

The CNV analysis conducted by Ginkgo is subjected to BSD-2-Clause license.
```


