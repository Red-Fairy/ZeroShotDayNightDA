"""
This script is used to check 
whether the mistakenly transformed avi videos all match to their existing original file
"""

import os

def check_avi(dir_path):
        
    avi_list = []
    mp4_list = []
    missing_list = []


    for file in os.listdir(dir_path):
        if (os.path.splitext(file)[1]=='.avi'):
        	avi_list.append(file.split(".")[0])
        elif (os.path.splitext(file)[1]=='.mp4'):
        	mp4_list.append(file.split(".")[0])

    for avi_file in avi_list:
    	if avi_file in mp4_list:
		print(avi_file)
		continue
    	else:
    		print("The original mp4 videos of %s does not exist" %(file))
    		missing_list.append(avi_file)

    if missing_list == []:
    	print("All original videos of avi clips are available")

if __name__ == '__main__':
	check_avi(os.getcwd())
