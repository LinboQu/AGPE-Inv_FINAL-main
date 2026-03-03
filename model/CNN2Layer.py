'''
CNN style model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	'''
		input： 是地震1D波形
		output： 反射系数

		模型结构： 2*[ conv_1D(1*80) + ReLU ]
	'''
	def __init__(self):
		super(CNN, self).__init__()

		self.noOfNeurons = 60   # channel,特征维
		self.dilation = 1
		self.kernel_size = 80   # 卷积核
		self.stride = 1
		self.padding = int(((self.dilation*(self.kernel_size-1)-1)/self.stride-1)/2)  # same卷积

		self.layer1 = nn.Sequential(
			nn.Conv1d(1, self.noOfNeurons, kernel_size=self.kernel_size, stride=1, padding=self.padding+1),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv1d(self.noOfNeurons, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding+2),
			nn.ReLU())

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)

		return out


class VishalNet_par(nn.Module):
	'''
		模型结构： 
					Conv_1d(1*81) + ReLU + Conv_1d(1*301) ---
															 |--cat -- conv1d --out
					Conv_1d(1*9) +  ReLU + Conv_1d(1*9) -----
	'''
	def __init__(self):
		super(VishalNet_par, self).__init__()
		self.cnn1 = nn.Conv1d(1, 60, 9, 1, 4)    # 卷积核为81, padding=(k-1)/2
		self.cnn2 = nn.Conv1d(60, 1, 9, 1, 4)    # 卷积核为301
		self.ReLU1 = nn.ReLU()
		self.ReLU2 = nn.ReLU()

		self.cnn3 = nn.Conv1d(1, 60, 81, 1, 40)    # 卷积核为81, padding=(k-1)/2
		self.cnn4 = nn.Conv1d(60, 1, 301, 1, 150)    # 卷积核为301		
		self.ReLU3 = nn.ReLU()
		self.ReLU4 = nn.ReLU()

		self.cnn_out1 = nn.Conv1d(2, 1, 15, 1, 7)

	
	def forward(self, input):
		out1 = self.ReLU1(self.cnn1(input))
		out1 = self.ReLU2(self.cnn2(out1))

		out2 = self.ReLU3(self.cnn3(input))
		out2 = self.ReLU4(self.cnn4(out2))

		# 两个支路的特征加和效果没有cat效果好。。。
		out = torch.cat([out1, out2], dim=1)
		# out = out1 + out2

		out = self.cnn_out1(out)

		return out

#################  1-D Physics-based Model by Vishal Das et al. ###################################################
class VishalNet(nn.Module):
	'''
		模型结构：
			主干：Conv_1d(1*81) + ReLU + Conv_1d(1*301)
			细节支路（可选）：在高通输入上做轻量卷积，输出残差细节
			输出：main + detail_gain * detail
	'''
	def __init__(
		self,
		input_dim=1,
		use_detail_branch=True,
		detail_gain=0.15,
		detail_hp_kernel=9,
		detail_channels=24,
		detail_dilations=(1, 2, 4),
		detail_kernel_sizes=(9, 7, 5),
	):
		super(VishalNet, self).__init__()
		self.cnn1 = nn.Conv1d(input_dim, 60, 81, 1, 40)    # 卷积核为81, padding=(k-1)/2
		self.cnn2 = nn.Conv1d(60, 1, 301, 1, 150)  # 卷积核为301
		self.use_detail_branch = bool(use_detail_branch)
		self.detail_gain = float(detail_gain)
		self.detail_hp_kernel = int(detail_hp_kernel)

		dils = list(detail_dilations) if detail_dilations is not None else [1, 2, 4]
		ks = list(detail_kernel_sizes) if detail_kernel_sizes is not None else [9, 7, 5]
		if len(dils) < 2:
			dils = [1, 2, 4]
		if len(ks) < 2:
			ks = [9, 7, 5]
		if len(ks) < len(dils):
			ks = ks + [ks[-1]] * (len(dils) - len(ks))
		if len(dils) < len(ks):
			dils = dils + [dils[-1]] * (len(ks) - len(dils))

		if self.use_detail_branch:
			ch = int(detail_channels) if int(detail_channels) > 0 else 24
			k1, d1 = int(ks[0]), int(dils[0])
			k2, d2 = int(ks[1]), int(dils[1])
			if len(ks) >= 3:
				k3, d3 = int(ks[2]), int(dils[2])
			else:
				k3, d3 = k2, d2
			self.detail_conv1 = nn.Conv1d(input_dim, ch, k1, 1, (k1 // 2) * d1, dilation=d1)
			self.detail_conv2 = nn.Conv1d(ch, ch, k2, 1, (k2 // 2) * d2, dilation=d2)
			self.detail_conv3 = nn.Conv1d(ch, 1, k3, 1, (k3 // 2) * d3, dilation=d3)
	
	def forward(self, input):
		out1 = F.relu(self.cnn1(input))
		out2 = self.cnn2(out1)

		# 兼容旧模型对象：若没有细节分支属性则自动退化为原始VishalNet。
		use_detail = bool(getattr(self, "use_detail_branch", False))
		has_detail = hasattr(self, "detail_conv1") and hasattr(self, "detail_conv2") and hasattr(self, "detail_conv3")
		if use_detail and has_detail:
			hp = input
			k = int(getattr(self, "detail_hp_kernel", 0))
			if (k >= 3) and ((k % 2) == 1):
				lp = F.avg_pool1d(hp, kernel_size=k, stride=1, padding=k // 2)
				hp = hp - lp
			d = F.relu(self.detail_conv1(hp))
			d = F.relu(self.detail_conv2(d))
			d = self.detail_conv3(d)
			g = float(getattr(self, "detail_gain", 0.0))
			out2 = out2 + g * d
		return out2

class CNN_R(nn.Module):
	'''
		结构： 2*[conv_1D(1*80) + ReLU] + conv_1D(1*1)
	'''
	def __init__(self):
		super(CNN_R, self).__init__()

		self.noOfNeurons = 60   # channel,特征维
		self.dilation = 1
		self.kernel_size = 80   # 卷积核
		self.stride = 1
		self.padding = int(((self.dilation*(self.kernel_size-1)-1)/self.stride-1)/2)  # same卷积

		self.layer1 = nn.Sequential(
			nn.Conv1d(1, self.noOfNeurons, kernel_size=self.kernel_size, stride=1, padding=self.padding+1),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv1d(self.noOfNeurons, 3, kernel_size=self.kernel_size, stride=1, padding=self.padding+2),
			nn.ReLU())

		# self.layer3 = nn.Conv1d(3, 1, kernel_size=3, padding=1)
		self.layer3 = nn.Conv1d(3, 1, kernel_size=1, padding=0)  # 拟合的形式

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)

		return out
