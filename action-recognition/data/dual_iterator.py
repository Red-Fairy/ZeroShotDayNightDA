"""
Author: Xu Yuecong
"""
import os
import cv2
import numpy as np

import torch.utils.data as data
import logging

from .flow_iterator import Flow
from .video_iterator import Video

class DualIter(data.Dataset):

    def __init__(self,
                 video_prefix,
                 flow_prefix,
                 txt_list,
                 video_sampler,
                 flow_sampler,
                 video_transforms=None,
                 flow_transforms=None,
                 name="<NO_NAME>",
                 force_color=True,
                 force_gray=True,
                 cached_info_path=None,
                 return_item_subpath=False,
                 shuffle_list_seed=None,
                 check_video=False,
                 tolerant_corrupted_video=None):
        super(DualIter, self).__init__()
        # load params
        self.video_sampler = video_sampler
        self.flow_sampler = flow_sampler
        self.force_color = force_color
        self.force_gray = force_gray
        self.video_prefix = video_prefix
        self.flow_prefix = flow_prefix
        self.video_prefix_x = os.path.join(flow_prefix, 'flow_x')
        self.video_prefix_y = os.path.join(flow_prefix, 'flow_y')
        self.video_transforms = video_transforms
        self.flow_transforms = flow_transforms
        self.return_item_subpath = return_item_subpath
        self.backup_item = None
        if (not check_video) and (tolerant_corrupted_video is None):
            logging.warning("DualIter:: >> `check_video' is off, `tolerant_corrupted_video' is automatically activated.")
            tolerant_corrupted_video = True
        self.tolerant_corrupted_video = tolerant_corrupted_video
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(video_prefix=self.video_prefix,
                                               txt_list=txt_list,
                                               check_video=check_video,
                                               cached_info_path=cached_info_path)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("DualIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, index):
        # get current video info
        v_id, label, vid_subpath, frame_count = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)
        video_path_x = os.path.join(self.video_prefix_x, vid_subpath)
        video_path_y = os.path.join(self.video_prefix_y, vid_subpath)

        faulty_frames = []
        successfule_trial = False
        try:
            with Video(vid_path=video_path) as video:
                if frame_count < 0:
                    frame_count = video.count_frames(check_validity=False)
                for i_trial in range(20):
                    # dynamic sampling
                    sampled_idxs = self.video_sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
                    if not 'SegmentalSampling' in self.video_sampler.__class__.__name__:
                        sampled_idxs = list(sampled_idxs)
                    if set(sampled_idxs).intersection(faulty_frames):
                        continue
                    prev_sampled_idxs = sampled_idxs
                    # extracting frames
                    sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
                    if sampled_frames is None:
                        faulty_frames.append(video.faulty_frame)
                    else:
                        successfule_trial = True
                        break
            with Flow(vid_path=video_path_x) as video_x:
                if frame_count < 0:
                    frame_count = video_x.count_frames(check_validity=False)
                for i_trial in range(20):
                    # dynamic sampling
                    sampled_idxs_x = self.flow_sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
                    if not 'SegmentalSampling' in self.flow_sampler.__class__.__name__:
                        sampled_idxs_x = list(sampled_idxs_x)
                    if set(sampled_idxs_x).intersection(faulty_frames):
                        continue
                    prev_sampled_idxs = sampled_idxs_x
                    # extracting frames
                    sampled_frames_x = video_x.extract_frames(idxs=sampled_idxs_x, force_gray=self.force_gray)
                    if sampled_frames_x is None:
                        faulty_frames.append(video_x.faulty_frame)
                    else:
                        successfule_trial = True
                        with Flow(vid_path=video_path_y) as video_y:
                            sampled_frames_y = video_y.extract_frames(idxs=sampled_idxs_x, force_gray=self.force_gray)
                        break
        except IOError as e:    
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))

        if not successfule_trial:
            assert (self.backup_item is not None), \
                "DualIter:: >> frame {} is error & backup is inavailable. [{}]'".format(faulty_frames, video_path_x)
            # logging.warning(">> frame {} is error, use backup item! [{}]".format(faulty_frames, video_path_x))
            with Video(vid_path=self.backup_item['video_path']) as video:
                sampled_frames = video.extract_frames(idxs=self.backup_item['sampled_idxs'], force_color=self.force_color)
            with Flow(vid_path=self.backup_item['video_path_x']) as video_x:
                sampled_frames_x = video_x.extract_frames(idxs=self.backup_item['sampled_idxs_x'], force_gray=self.force_gray)
            with Flow(vid_path=self.backup_item['video_path_y']) as video_y:
                sampled_frames_y = video_y.extract_frames(idxs=self.backup_item['sampled_idxs_y'], force_gray=self.force_gray)            
        elif self.tolerant_corrupted_video:
            # assume the error rate less than 10%
            if (self.backup_item is None) or (self.rng.rand() < 0.1):
                self.backup_item = {'video_path': video_path, 'video_path_x': video_path_x, 'video_path_y': video_path_y, 
                                    'sampled_idxs': sampled_idxs, 'sampled_idxs_x': sampled_idxs_x, 'sampled_idxs_y': sampled_idxs_x}

        sampled_frames_flow = []
        for idx in range(len(sampled_frames_x)):
            sampled_frames_flow.append(np.stack([sampled_frames_x[idx], sampled_frames_y[idx]], axis=2))
        clip_input = np.concatenate(sampled_frames, axis=2)
        clip_input_flow = np.concatenate(sampled_frames_flow, axis=2)
        # apply video augmentation
        if self.video_transforms is not None:
            clip_input = self.video_transforms(clip_input)
        if self.flow_transforms is not None:
            clip_input_flow = self.flow_transforms(clip_input_flow)
        return clip_input, clip_input_flow, label, vid_subpath


    def __getitem__(self, index):
        succ = False
        attempts = 0
        while not succ and attempts < 5:
            try:
                clip_input, clip_input_flow, label, vid_subpath = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                attempts = attempts + 1
                logging.warning("DualIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        if self.return_item_subpath:
            return clip_input, clip_input_flow, label, vid_subpath
        else:
            return clip_input, clip_input_flow, label


    def __len__(self):
        return len(self.video_list)


    def _get_video_list(self,
                        video_prefix,
                        txt_list,
                        check_video=False,
                        cached_info_path=None):
        # formate:
        # [vid, label, video_subpath, frame_count]
        assert os.path.exists(video_prefix), "DualIter:: failed to locate: `{}'".format(video_prefix)
        assert os.path.exists(txt_list), "DualIter:: failed to locate: `{}'".format(txt_list)

        # try to load cached list
        cached_video_info = {}
        if cached_info_path:
            if os.path.exists(cached_info_path):
                f = open(cached_info_path, 'r')
                cached_video_prefix = f.readline().split()[1]
                cached_txt_list = f.readline().split()[1]
                if (cached_video_prefix == video_prefix.replace(" ", "")) \
                    and (cached_txt_list == txt_list.replace(" ", "")):
                    logging.info("DualIter:: loading cached video info from: `{}'".format(cached_info_path))
                    lines = f.readlines()
                    for line in lines:
                        video_subpath, frame_count = line.split()
                        cached_video_info.update({video_subpath: int(frame_count)})
                else:
                    logging.warning(">> Cached video list mismatched: " +
                                    "(prefix:{}, list:{}) != expected (prefix:{}, list:{})".format(\
                                    cached_video_prefix, cached_txt_list, video_prefix, txt_list))
                f.close()
            else:
                if not os.path.exists(os.path.dirname(cached_info_path)):
                    os.makedirs(os.path.dirname(cached_info_path))

        # building dataset
        video_list = []
        new_video_info = {}
        logging_interval = 100
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("DualIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_path_x = os.path.join(video_prefix, video_subpath)
                if not os.path.exists(video_path_x):
                    # logging.warning("DualIter:: >> cannot locate `{}'".format(video_path_x))
                    continue
                if check_video:
                    if video_subpath in cached_video_info:
                        frame_count = cached_video_info[video_subpath]
                    elif video_subpath in new_video_info:
                        frame_count = cached_video_info[video_subpath]
                    else:
                        frame_count = self.video.open(video_path_x).count_frames(check_validity=True)
                        new_video_info.update({video_subpath: frame_count})
                else:
                    frame_count = -1
                info = [int(v_id), int(label), video_subpath, frame_count]
                video_list.append(info)
                if check_video and (i % logging_interval) == 0:
                    logging.info("DualIter:: - Checking: {:d}/{:d}, \tinfo: {}".format(i, len(lines), info))

        # caching video list
        if cached_info_path and len(new_video_info) > 0:
            logging.info("DualIter:: adding {} lines new video info to: {}".format(len(new_video_info), cached_info_path))
            cached_video_info.update(new_video_info)
            with open(cached_info_path, 'w') as f:
                # head
                f.write("video_prefix: {:s}\n".format(video_prefix.replace(" ", "")))
                f.write("txt_list: {:s}\n".format(txt_list.replace(" ", "")))
                # content
                for i, (video_subpath, frame_count) in enumerate(cached_video_info.items()):
                    if i > 0:
                        f.write("\n")
                    f.write("{:s}\t{:d}".format(video_subpath, frame_count))

        return video_list