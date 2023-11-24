"""
Author: Xu Yuecong
"""
import os
import cv2
import numpy as np

import torch.utils.data as data
import logging


class Flow(object):
    """basic Flow class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path), "FlowIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("FlowIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self, check_validity=False):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
        if check_validity:
            verified_frame_count = 0
            for i in range(unverified_frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if not self.cap.grab():
                    logging.warning("FlowIter:: >> frame (start from 0) {} corrupted in {}".format(i, self.vid_path))
                    break
                verified_frame_count = i + 1
            self.frame_count = verified_frame_count
        else:
            self.frame_count = unverified_frame_count
        assert self.frame_count > 0, "FlowIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count

    def extract_frames(self, idxs, force_gray=True):
        frames = self.extract_frames_fast(idxs, force_gray)
        if frames is None:
            # try slow method:
            frames = self.extract_frames_slow(idxs, force_gray)
        return frames

    def extract_frames_fast(self, idxs, force_gray=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read() # in BGR/GRAY format
            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) >= 3:
                if force_gray:
                    # Convert BGR to Gray
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        return frames

    def extract_frames_slow(self, idxs, force_gray=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = [None] * len(idxs)
        idx = min(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while idx <= max(idxs):
            res, frame = self.cap.read() # in BGR/GRAY format
            if not res:
                # end of the video
                self.faulty_frame = idx
                return None
            if idx in idxs:
                # fond a frame
                if len(frame.shape) >= 3:
                    if force_gray:
                        # Convert BGR to Gray
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pos = [k for k, i in enumerate(idxs) if i == idx]
                for k in pos:
                    frames[k] = frame
            idx += 1
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self


class FlowIter(data.Dataset):

    def __init__(self,
                 video_prefix,
                 txt_list,
                 sampler,
                 flow_transforms=None,
                 name="<NO_NAME>",
                 force_gray=True,
                 cached_info_path=None,
                 return_item_subpath=False,
                 shuffle_list_seed=None,
                 check_video=False,
                 tolerant_corrupted_video=None):
        super(FlowIter, self).__init__()
        # load params
        self.sampler = sampler
        self.force_gray = force_gray
        self.video_prefix = video_prefix
        self.video_prefix_x = os.path.join(video_prefix, 'flow_x')
        self.video_prefix_y = os.path.join(video_prefix, 'flow_y')
        self.flow_transforms = flow_transforms
        self.return_item_subpath = return_item_subpath
        self.backup_item = None
        if (not check_video) and (tolerant_corrupted_video is None):
            logging.warning("FlowIter:: >> `check_video' is off, `tolerant_corrupted_video' is automatically activated.")
            tolerant_corrupted_video = True
        self.tolerant_corrupted_video = tolerant_corrupted_video
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(video_prefix_x=self.video_prefix_x,
                                               txt_list=txt_list,
                                               check_video=check_video,
                                               cached_info_path=cached_info_path)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("FlowIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, index):
        # get current video info
        v_id, label, vid_subpath, frame_count = self.video_list[index]
        video_path_x = os.path.join(self.video_prefix_x, vid_subpath)
        video_path_y = os.path.join(self.video_prefix_y, vid_subpath)

        faulty_frames = []
        successfule_trial = False
        try:
            with Flow(vid_path=video_path_x) as video:
                if frame_count < 0:
                    frame_count = video.count_frames(check_validity=False)
                for i_trial in range(20):
                    # dynamic sampling
                    sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
                    if not 'SegmentalSampling' in self.sampler.__class__.__name__:
                        sampled_idxs = list(sampled_idxs)
                    if set(sampled_idxs).intersection(faulty_frames):
                        continue
                    prev_sampled_idxs = sampled_idxs
                    # extracting frames
                    sampled_frames_x = video.extract_frames(idxs=sampled_idxs, force_gray=self.force_gray)
                    if sampled_frames_x is None:
                        faulty_frames.append(video.faulty_frame)
                    else:
                        successfule_trial = True
                        with Flow(vid_path=video_path_y) as video_y:
                            sampled_frames_y = video_y.extract_frames(idxs=sampled_idxs, force_gray=self.force_gray)
                        break
        except IOError as e:
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))

        if not successfule_trial:
            assert (self.backup_item is not None), \
                "FlowIter:: >> frame {} is error & backup is inavailable. [{}]'".format(faulty_frames, video_path_x)
            # logging.warning(">> frame {} is error, use backup item! [{}]".format(faulty_frames, video_path_x))
            with Flow(vid_path=self.backup_item['video_path_x']) as video:
                sampled_frames_x = video.extract_frames(idxs=self.backup_item['sampled_idxs'], force_gray=self.force_gray)
            with Flow(vid_path=self.backup_item['video_path_y']) as video_y:
                sampled_frames_y = video_y.extract_frames(idxs=self.backup_item['sampled_idxs'], force_gray=self.force_gray)            
        elif self.tolerant_corrupted_video:
            # assume the error rate less than 10%
            if (self.backup_item is None) or (self.rng.rand() < 0.1):
                self.backup_item = {'video_path_x': video_path_x, 'video_path_y': video_path_y, 'sampled_idxs': sampled_idxs}

        sampled_frames = []
        for idx in range(len(sampled_frames_x)):
            sampled_frames.append(np.stack([sampled_frames_x[idx], sampled_frames_y[idx]], axis=2))
        clip_input = np.concatenate(sampled_frames, axis=2)
        # apply video augmentation
        if self.flow_transforms is not None:
            clip_input = self.flow_transforms(clip_input)
        return clip_input, label, vid_subpath


    def __getitem__(self, index):
        succ = False
        attempts = 0
        while not succ and attempts < 5:
            try:
                clip_input, label, vid_subpath = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("FlowIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))
                attempts = attempts + 1

        if self.return_item_subpath:
            return clip_input, label, vid_subpath
        else:
            return clip_input, label


    def __len__(self):
        return len(self.video_list)


    def _get_video_list(self,
                        video_prefix_x,
                        txt_list,
                        check_video=False,
                        cached_info_path=None):
        # formate:
        # [vid, label, video_subpath, frame_count]
        assert os.path.exists(video_prefix_x), "FlowIter:: failed to locate: `{}'".format(video_prefix_x)
        assert os.path.exists(txt_list), "FlowIter:: failed to locate: `{}'".format(txt_list)

        # try to load cached list
        cached_video_info = {}
        if cached_info_path:
            if os.path.exists(cached_info_path):
                f = open(cached_info_path, 'r')
                cached_video_prefix_x = f.readline().split()[1]
                cached_txt_list = f.readline().split()[1]
                if (cached_video_prefix_x == video_prefix_x.replace(" ", "")) \
                    and (cached_txt_list == txt_list.replace(" ", "")):
                    logging.info("FlowIter:: loading cached video info from: `{}'".format(cached_info_path))
                    lines = f.readlines()
                    for line in lines:
                        video_subpath, frame_count = line.split()
                        cached_video_info.update({video_subpath: int(frame_count)})
                else:
                    logging.warning(">> Cached video list mismatched: " +
                                    "(prefix:{}, list:{}) != expected (prefix:{}, list:{})".format(\
                                    cached_video_prefix_x, cached_txt_list, video_prefix_x, txt_list))
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
            logging.info("FlowIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_path_x = os.path.join(video_prefix_x, video_subpath)
                if not os.path.exists(video_path_x):
                    # logging.warning("FlowIter:: >> cannot locate `{}'".format(video_path_x))
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
                    logging.info("FlowIter:: - Checking: {:d}/{:d}, \tinfo: {}".format(i, len(lines), info))

        # caching video list
        if cached_info_path and len(new_video_info) > 0:
            logging.info("FlowIter:: adding {} lines new video info to: {}".format(len(new_video_info), cached_info_path))
            cached_video_info.update(new_video_info)
            with open(cached_info_path, 'w') as f:
                # head
                f.write("video_prefix_x: {:s}\n".format(video_prefix_x.replace(" ", "")))
                f.write("txt_list: {:s}\n".format(txt_list.replace(" ", "")))
                # content
                for i, (video_subpath, frame_count) in enumerate(cached_video_info.items()):
                    if i > 0:
                        f.write("\n")
                    f.write("{:s}\t{:d}".format(video_subpath, frame_count))

        return video_list
