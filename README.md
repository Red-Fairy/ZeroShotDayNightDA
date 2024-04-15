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
- 09/06/2023: Code for image classification and semantic segmentation is available now.
- 10/18/2023: Code for visual place recognition is available now.
- 11/25/2023: Code for video action recognition is available now.

## Code
Code for image classification and semantic segmentation is available now. Code for visual place recognition and video action recognition will be released soon. 

### Requirements
**Environment**: Pytorch with version >= 1.11.0 is required. Other requirements can be easily satisfied using `pip install`.

**GPU**: 3 GPUs with at least 12GB memory (e.g., 2080Ti) are required.

### Image Classification
#### Dataset Preparation
Download the [CODaN dataset](https://github.com/Attila94/CIConv) and put it under `./classification/data/`.

#### Training
- Navigate to `./darkening`, run `python darken_classification.py --sim --experiment EXPERIMENT_NAME` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--sim_model_dir`. Specify the logging directory by `--experiment`, and models will be saved under `./darkening/checkpoints/{args.experiment}`.
- Navigate to `./classification`, run `python train.py --use_BYOL` using the BYOL loss ($\mathcal{L}_F^{sim}$). Specify the pre-trained darkening model path with `--darkening_model`, the pre-trained daytime model with `--checkpoint`, and the logging directory by `--experiment`. Model checkpoints and loggers will be saved under `./classification/checkpoints/{args.experiment}`.
- Classification results will be saved in `./classification/results/{args.experiment}/log.txt`. You may also run ``python test.py`` to test the model.

### Semantic Segmentation
#### Dataset Preparation
We need three dataset for training and evaluation: [Cityscapes](https://www.cityscapes-dataset.com/), [Nighttime Driving](http://people.ee.ethz.ch/~daid/NightDriving/#), and [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/). Download and put them under `./segmentation/data/`.

#### Training
- Navigate to `./darkening`, run `python darken_segmentation.py --sim --experiment EXPERIMENT_NAME` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--sim_model_dir`, the logging directory by `--experiment`, and models will be saved under `./darkening/checkpoints/{args.experiment}`.
- To save GPU memory, our implementation generates the darkened nighttime dataset in advance. Run python `darken_test.py` and specify the source daytime dataset path with `--src_path`, the darkening model path with `--experiment`, and the target nighttime dataset path with `--output_dir`. The darkened nighttime dataset will be saved under `--dst_path`. You may also download our pre-generated darkened nighttime dataset [here](https://drive.google.com/file/d/1b9KVLWpTpY1yVhA7j5nrcHPizZkgLz36/view?usp=sharing).
- Navigate to `./segmentation`, run `python train.py`. Specify the darkened dataset by `--darken_dataset` and the logging directory by `--experiment`. Model checkpoints and loggers will be saved under `./segmentation/runs/{args.experiment}`.
- Segmentation results will be saved in `./segmentation/runs/{args.experiment}/logs/`. You may also run ``python eval_test.py`` to obtain the visualization results and the zipped file for Dark Zurich evaluation.

### Visual Place Recognition
#### Dataset Preparation
- Please modify the return value of the function `get_data_root` in `./visual-place-recognition/cirtorch/utils/general.py` to the path of the dataset. The datasets will be automatically downloaded and extracted. You may also download the dataset from [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/) and put it under `./visual-place-recognition/dataset/`. The training data will be stored in `./retrieval/train/retrieval-SfM-120k`. 

#### Training
- Navigate to `./darkening` run `python darken_vpr.py` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--sim_model_dir`, the logging directory by `--experiment`, and models will be saved under `./darkening/checkpoints/{args.experiment}`.
- Navigate to `./visual-place-recognition`, run `python3 -m cirtorch.examples.train_night EXPERIMENT_NAME --training-dataset 'retrieval-SfM-120k'  --test-datasets '247tokyo1k' --arch 'resnet101' --pool 'gem' --loss 'contrastive'  --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=22000 --pool-size=2000 --batch-size 5 --image-size 362 --epochs 5 --darkening_model PATH_TO_DARKENING_MODEL --sim`. 
- Model checkpoints and loggers will be saved under `./visual-place-recognition/checkpoints/EXPERIMENT_NAME`. Run `python3 -m cirtorch.examples.test --network-path PATH_TO_CHECKPOINT --datasets '247tokyo1k' --whitening 'retrieval-SfM-120k' --multiscale '[1, 1/2**(1/2), 1/2]'` to test the model.

### Low-Light Action Recognition
#### Dataset Preparation
- Download the normal light data from [here](https://drive.google.com/drive/folders/1iG3VwUuAXZFofE0tciYhkEGfr49WGmd1) (CVPR'22 UG2 challenge). It contains 2,625 videos from 11 classes. Put it under `./low-light-action-recognition/dataset/NormalLight`.
- Download the low light data (ARID dataset) from [here](https://drive.google.com/file/d/10sitw9Mi9Gv1jMfyMwbv78EZSpW_lKEx/view?usp=sharing). Put it under `./low-light-action-recognition/dataset/ARID`.

#### Training
- Navigate to `./low-light-action-recognition`, run `python train_darkening.py --sim --experiment EXPERIMENT_NAME` to train the darkening model with the $\mathcal{L}_D^{sim}$. Specify the pre-trained daytime model path with `--feature_extractor` (download from our pretrained model link), the logging directory by `--experiment`, and models will be saved under `./low-light-action-recognition/checkpoints/{args.experiment}`.
- Run `darken_video.py --darkening_model PATH_TO_DARKENING_MODEL --pretrained_model_path PATH_TO_DAYTIME_PRETRAINED_MODEL`. The darkened videos will be saved under the specified directory (use `./dataset/NormalLight/raw/data_darken_test/` by default, otherwise you should also change the darkened data path under `./data/iterator_factory`).
- Run `train_night.py --model-dir SAVE_DIR --load_checkpoint PATH_TO_PRETRAINED_DAYTIME_MODEL`. Model checkpoints and loggers will be saved under `./low-light-action-recognition/checkpoints/{args.model_dir}`.
- Run `test.py --model-dir PATH_TO_CHECKPOINT` for evaluation.

### Pre-trained Models
We provide all the data and pre-trained models [here](
https://drive.google.com/drive/folders/1E1BizpMh-G-eJVISHJ0mw8717ryAnul3?usp=sharing).



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
