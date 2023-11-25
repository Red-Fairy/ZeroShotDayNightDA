import os
import cv2
import sys
import glob
import numpy as np
import subprocess

target_folder = 'ori_frames'
root_folder = 'avi'
i = 1
total_files = sum([len(files) for r, d, files in os.walk(root_folder)])

if not os.path.exists(target_folder):
	os.mkdir(target_folder)

for dir in os.listdir(root_folder):
	class_folder = os.path.join(root_folder, dir)
	target_class_folder = os.path.join(target_folder, dir)
	if not os.path.exists(target_class_folder):
		os.mkdir(target_class_folder)
	files = glob.glob(os.path.join(class_folder, '*.avi'))
	for file in files:
		print("Processing {}".format(file))
		filename = file.split('/')[-1]
		output_folder = os.path.join(target_class_folder, filename)
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)

		cap = cv2.VideoCapture(clip_path)
		idx = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
				break
			output_frame = 'img_{:05d}'.format(idx) + '.jpg'
			output_file = os.path.join(output_folder, output_frame)
			cv2.imwrite(output_file, frame)
			idx = idx + 1

		cap.release()
		cv2.destroyAllWindows()		
		print("Process {} completed ({}/{})".format(file, i, total_files))
		i = i + 1