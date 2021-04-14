# Deep Learning-based Building Footprint Extraction with Missing Annotations

[Jian Kang](https://github.com/jiankang1991), [Ruben Fernandez-Beltran](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es), [Xian Sun](http://people.ucas.ac.cn/~sunxian), [Jingen Ni](https://scholar.google.com/citations?hl=en&user=hqZB5wQAAAAJ&view_op=list_works&sortby=pubdate), [Antonio Plaza](https://www.umbc.edu/rssipl/people/aplaza/)

---

This repo contains the codes for the GRSL paper: [Deep Learning-based Building Footprint Extraction with Missing Annotations](). We propose a novel loss function for extracting building footprints based on the training images with missing annotations. The loss includes three terms: 1) logit adjusted cross entropy (LACE) loss, aimed at discriminating between building and background pixels from a long-tailed label distribution; 2) weighted dice loss, aimed at increasing the F1 scores of the predicted building masks; and 3) boundary alignment loss, which is optimized for preserving the fine-grained structure of building boundaries.

<p align="center">
<img src="./pic/pic1.png" alt="drawing" width="300"/>
</p>

<p align="center">
<img src="./pic/pic2.png" alt="drawing"/>
</p>

## Usage

`train/main_.py` is the script of the proposed method for training and validation.

`utils/loss_.py` contains the considered losses for building footprint extraction.

Some codes are modified from [SegLoss](https://github.com/JunMa11/SegLoss) and [Geoseg](https://github.com/huster-wgm/geoseg). 

## Citation

```
@article{kang2021bdsegma,
  title={{Deep Learning-based Building Footprint Extraction with Missing Annotations}},
  author={Kang, Jian and Fernandez-Beltran, Ruben and Sun, Xian and Ni, Jingen and Plaza, Antonio},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2021},
  note={DOI:10.1109/LGRS.2021.3072589}
  publisher={IEEE}
}
```


