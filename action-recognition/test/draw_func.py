import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import seaborn as sns


def draw_hist(hist, save_name):
	hist = hist.squeeze(0).data.cpu().numpy()
	plt.figure(figsize=(3.3, 3))
	# sns.set_theme(style="white")
	plt.subplots_adjust(left=0.2)
	plt.rcParams['xtick.direction'] = 'in'
	plt.rcParams['ytick.direction'] = 'in'

	for i,c in zip([0,1,2],['r','g','b']):
		single_hist = hist[i,:]
		x = np.array(range(len(single_hist)))/255.
		plt.scatter(x, single_hist, color=c, s=1)

	plt.xticks(np.arange(0, 1.01, 0.25))
	plt.yticks(np.arange(0, 1.01, 0.25))
	plt.savefig(save_name)
	plt.close()


def draw_hist_grad(hist, save_name):
	hist = hist.squeeze(0).data.cpu().numpy()
	plt.figure(figsize=(3.3, 3))
	# sns.set_theme(style="white")
	plt.subplots_adjust(left=0.2)
	plt.rcParams['xtick.direction'] = 'in'
	plt.rcParams['ytick.direction'] = 'in'

	for i,c in zip([0,1,2],['r','g','b']):
		single_hist = hist[i,:]
		x = np.array(range(len(single_hist)))/255.
		plt.scatter(x, single_hist, color=c, s=1)

	plt.xticks(np.arange(0, 1.01, 0.25))
	# plt.yticks(np.arange(0, 1.01, 0.25))
	plt.savefig(save_name)
	plt.close()


# def draw_point(data_list, model_type, save_name):
# 	for i,c in zip([0,1,2],['r','g','b']):
# 		for data in data_list[i]:
# 			if model_type == 'II' or model_type == 'II-AVG': # 1/x
# 				a = data[0]
# 				x = data[1]
# 				z = (a+1) * x / (x+a+1e-10)
# 				plt.scatter(x, z, s=2, c=c)
		
# 			elif model_type == 'III' or model_type == 'III-AVG': # ax^2 + b^2 + c, multiple times
# 				a1 = data[0]
# 				a2 = data[1]
# 				a3 = data[2]
# 				a4 = data[3]
# 				a5 = data[4]
# 				a6 = data[5]
# 				a7 = data[6]
# 				a8 = data[7]
# 				x = data[8]

# 				z = x + a1 * (x ** 2 - x)
# 				z = z + a2 * (z ** 2 - z)
# 				z = z + a3 * (z ** 2 - z)
# 				z = z + a4 * (z ** 2 - z)
# 				z = z + a5 * (z ** 2 - z)
# 				z = z + a6 * (z ** 2 - z)
# 				z = z + a7 * (z ** 2 - z)
# 				z = z + a8 * (z ** 2 - z)
# 				plt.scatter(x, z, s=2, c=c)

# 			elif model_type == 'IV' or model_type == 'IV-AVG': # a ^ x
# 				a = data[0]
# 				x = data[1]
# 				z = (a ** x - 1) / (a-1)
# 				plt.scatter(x, z, s=2, c=c)

# 			elif model_type == 'V' or model_type == 'V-AVG': # x ^ a
# 				a = data[0]
# 				x = data[1]
# 				z = x ** a
# 				plt.scatter(x, z, s=2, c=c)

# 			elif model_type == 'V-Plus' or model_type == 'V-Plus-AVG': # x ^ a
# 				a, alpha1, alpha2 = data[:-1]
# 				x = data[-1]
# 				a2 = np.maximum(alpha1, alpha2) + 1e-1
# 				a1 = np.minimum(alpha1, alpha2)
# 				z = (((a2-a1)*x+a1) ** a- a1** a) / (a2 ** a - a1**a)
# 				plt.scatter(x, z, s=2, c=c)

# 			elif model_type == 'VI' or model_type == 'VI-AVG': # tan(x)
# 				a1 = data[0]
# 				a2 = data[1]
# 				x = data[2]
# 				z = (np.tan((a2-a1)*x+a1) - np.tan(a1)) / (np.tan(a2) - np.tan(a1))
# 				plt.scatter(x, z, s=2, c=c)

# 			elif model_type == 'VII' or model_type == 'VII-AVG': # log(x)
# 				a = data[0]
# 				x = data[1]
# 				z = np.log(a*x+1) / np.log(a+1)
# 				plt.scatter(x, z, s=2, c=c)

# 	plt.xlim((0, 1))
# 	plt.ylim((0, 1))

# 	plt.savefig(save_name)
# 	plt.close()


# def draw_line(data_list, model_type, save_name):
# 	x = np.arange(0, 1, 0.01)

# 	for data in data_list:
# 		if model_type == 'II' or model_type == 'II-AVG': # 1/x
# 			a = data[0]
# 			m = data[1] ** 0.25
# 			z = (a+1) * x / (x+a+1e-10)
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)
	
# 		elif model_type == 'III' or model_type == 'III-AVG': # ax^2 + b^2 + c, multiple times
# 			a1 = data[0]
# 			a2 = data[1]
# 			a3 = data[2]
# 			a4 = data[3]
# 			a5 = data[4]
# 			a6 = data[5]
# 			a7 = data[6]
# 			a8 = data[7]
# 			m = data[8] ** 0.25

# 			z = x + a1 * (x ** 2 - x)
# 			z = z + a2 * (z ** 2 - z)
# 			z = z + a3 * (z ** 2 - z)
# 			z = z + a4 * (z ** 2 - z)
# 			z = z + a5 * (z ** 2 - z)
# 			z = z + a6 * (z ** 2 - z)
# 			z = z + a7 * (z ** 2 - z)
# 			z = z + a8 * (z ** 2 - z)
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 		elif model_type == 'IV' or model_type == 'IV-AVG': # a ^ x
# 			a = data[0]
# 			m = data[1] ** 0.25
# 			z = (a ** x - 1) / (a-1)
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 		elif model_type == 'V' or model_type == 'V-AVG': # x ^ a
# 			a = data[0]
# 			m = data[1] ** 0.25
# 			z = x ** a
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 		elif model_type == 'V-Plus' or model_type == 'V-Plus-AVG': # x ^ a
# 			a, alpha1, alpha2 = data[:-1]
# 			m = data[-1] ** 0.25
# 			a2 = np.maximum(alpha1, alpha2) + 1e-1
# 			a1 = np.minimum(alpha1, alpha2)
# 			z = (((a2-a1)*x+a1) ** a- a1** a) / (a2 ** a - a1**a)
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 		elif model_type == 'VI' or model_type == 'VI-AVG': # tan(x)
# 			a1 = data[0]
# 			a2 = data[1]
# 			m = data[2] ** 0.25
# 			z = (np.tan((a2-a1)*x+a1) - np.tan(a1)) / (np.tan(a2) - np.tan(a1))
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 		elif model_type == 'VII' or model_type == 'VII-AVG': # log(x)
# 			a = data[0]
# 			m = data[1] ** 0.25
# 			z = np.log(a*x+1) / np.log(a+1)
# 			plt.plot(x, z, c=(m,0,1-m), linewidth=0.25)

# 	plt.xlim((0, 1))
# 	plt.ylim((0, 1))

# 	plt.savefig(save_name)
# 	plt.close()