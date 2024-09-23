# Genetic deconvolution of embryonic and maternal cell-free DNA in spent medium of human preimplantation embryos through deep learning


[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/zhenyiizhang/DECENT/blob/main/LICENSE)
[![commit](https://img.shields.io/github/last-commit/zhenyiizhang/DECENT?color=blue)](https://github.com/zhenyiizhang/DECENT/)


## Introduction
We present DECENT (**de**ep **C**NV r**e**co**n**s**t**ruction), a novel deep learning framework aimed at mitigating maternal contamination in spent culture embryo media (SECM) and reconstructing embryonic copy number variations (CNVs). DECENT leverages sequence and methylation information from both embryonic and maternal sources, utilizing convolutional neural networks and attention mechanisms to infer the origin of sequence reads. Overall, DECENT contributes to substantially enhancing the diagnostic accuracy and effectiveness of SECM-based niPGT, establishing a robust groundwork for the extensive clinical utilization of niPGT in the field of reproductive medicine.

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

3. If you want to conduct CNV analysis further, you should config [Ginkgo](https://github.com/robertaboukhalil/ginkgo) on your own server and then use our reference samples and scripts for analysis.



## Contact information

- Zhenyi Zhang-[zhenyizhang@stu.pku.edu.cn](mailto:zhenyizhang@stu.pku.edu.cn)
- Peijie Zhou (Corresponding author)-[pjzhou@pku.edu.cn](mailto:pjzhou@pku.edu.cn)

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


