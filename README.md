<h2 align="center">
  <b>Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation</b>

  <b><i>ICCV 2023 (Oral Presentation)</i></b>


<div align="center">
    <a href="https://github.com/Red-Fairy/ZeroShotDayNightDA" target="_blank">
    <img src="https://img.shields.io/badge/ICCV 2023-Oral Presentation-red"></a>
    <a href="https://arxiv.org/abs/2307.08779" target="_blank">
    <img src="https://img.shields.io/badge/Paper-orange" alt="paper"></a>
    <a href="https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/supp.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Supplementary-green" alt="supp"></a>
    <a href="https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-blue" alt="Project Page"/></a>
</div>
</h2>

---

This the official repository of the paper **Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation**.

For more information, please visit our [project website](https://red-fairy.github.io/ZeroShotDayNightDA-Webpage/).

**Authors:** Rundong Luo<sup>1</sup>, Wenjing Wang<sup>1</sup>, Wenhan Yang<sup>2</sup>, Jiaying Liu<sup>1</sup>

<sup>1</sup>Peking University, <sup>2</sup>Peng Cheng Laboratory

## Abstract
Low-light conditions not only hamper human visual experience but also degrade the model's performance on downstream vision tasks. While existing works make remarkable progress on day-night domain adaptation, they rely heavily on domain knowledge derived from the task-specific nighttime dataset. This paper challenges a more complicated scenario with border applicability, *i.e.*, zero-shot day-night domain adaptation, which eliminates reliance on any nighttime data. Unlike prior zero-shot adaptation approaches emphasizing either image-level translation or model-level adaptation, we propose a similarity min-max paradigm that considers them under a unified framework. On the image level, we darken images towards minimum feature similarity to enlarge the domain gap. Then on the model level, we maximize the feature similarity between the darkened images and their normal-light counterparts for better model adaptation. To the best of our knowledge, this work represents the pioneering effort in jointly optimizing both aspects, resulting in a significant improvement of model generalizability. Extensive experiments demonstrate our method's effectiveness and broad applicability on various nighttime vision tasks, including classification, semantic segmentation, visual place recognition, and video action recognition.

## Updates
- 09/05/2023: Code for image classification and semantic segmentation are available now.

## Code
Code for image classification and semantic segmentation is available now. Code for visual place recognition and video action recognition will be released soon. 

### Requirements
**Environment**: Pytorch with version >= 1.11.0 is required. Other requirements can be easily satisfied using `pip install`.

**GPU**: 3 GPUs with at least 12GB memory (e.g., 2080Ti) are required.

### Image Classification
#### Dataset Preparation
Download the [CODaN dataset](https://github.com/Attila94/CIConv) and put it under `./classification/data/`.

#### Training
- Navigate to `./darkening`, run `python darken_classification.py --sim` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--sim_model_dir`. Specify the logging directory by `--experiment`, and models will be saved under `./darkening/checkpoints/{args.experiment}`.
- Navigate to `./classification`, run `python train.py --use_BYOL` using the BYOL loss ($\mathcal{L}_F^{sim}$). Specify the pre-trained darkening model path with `--darkening_model`, the pre-trained daytime model with `--checkpoint`, and the logging directory by `--experiment`. Model checkpoints and loggers will be saved under `./classification/checkpoints/{args.experiment}`.
- Classification results will be saved in `./classification/results/{args.experiment}/log.txt`. You may also run ``python test.py`` to test the model.

### Semantic Segmentation
#### Dataset Preparation
We need three dataset for training and evaluation: [Cityscapes](https://www.cityscapes-dataset.com/), [Nighttime Driving](http://people.ee.ethz.ch/~daid/NightDriving/#), and [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/). Download and put them under `./segmentation/data/`.

#### Training
- Navigate to `./darkening`, run `python darken_segmentation.py --sim` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--sim_model_dir`, the logging directory by `--experiment`, and models will be saved under `./darkening/checkpoints/{args.experiment}`.
- To save GPU memory, our implementation generates the darkened nighttime dataset in advance. Run python `darken_test.py` and specify the source daytime dataset path with `--src_path`, the darkening model path with `--experiment`, and the target nighttime dataset path with `--output_dir`. The darkened nighttime dataset will be saved under `--dst_path`. You may also download our pre-generated darkened nighttime dataset [here](https://disk.pku.edu.cn:443/link/6B3418BCC0876977E2A4A56CA5568C78).
- Navigate to `./segmentation`, run `python train.py`. Specify the darkened dataset by `--darken_dataset` and the logging directory by `--experiment`. Model checkpoints and loggers will be saved under `./segmentation/runs/{args.experiment}`.
- Segmentation results will be saved in `./segmentation/runs/{args.experiment}/logs/`. You may also run ``python eval_test.py`` to obtain the visualization results and the zipped file for Dark Zurich evaluation.

### Pre-trained Models
We provide the pre-trained models for image classification and semantic segmentation, as well as the corresponding darkening models. You may download them [here](https://disk.pku.edu.cn:443/link/D12F2FAC207A60F4AB94197432B1032C).

## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{luo2023similarity,
  title={Similarity Min-Max: Zero-Shot Day-Night Domain Adaptation},
  author={Luo, Rundong and Wang, Wenjing and Yang, Wenhan and Liu, Jiaying},
  booktitle={ICCV},
  year={2023}
}
```

## Acknowledgement
Some code are borrowed from [CIConv](https://github.com/Attila94/CIConv). If you have any questions, please contact Rundong Luo [(rundongluo2002@gmail.com)](mailto:rundongluo2002@gmail.com) or open an issue.
