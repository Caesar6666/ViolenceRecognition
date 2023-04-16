import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as Data


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self,  txt, filepath='../Datasets/', n_frame=16):
        '''
        function: load video dataset
        :param txt: the list of video filepath
        :param n_frame: How many sample image from video wang to get
        '''
        fh = open(txt, 'r')  # 打开标签文件
        videos = []  # 创建列表，存储信息
        for line in fh:  # 遍历标签文件每行
            line = line.rstrip('\n')  # 删除字符串末尾的换行符
            words = line.split()  # 通过空格分割字符串，变成列表
            videos.append((words[0], int(words[1])))  # 把视频名words[0]，标签int(words[1])放到videos里
        fh.close()
        self.videos = videos
        self.n_frame = n_frame
        self.filepath = filepath

    def __getitem__(self, index):  # 检索函数
        fn, label = self.videos[index]  # 读取文件名、标签
        frames = self.get_frames(self.filepath+fn, self.n_frame)  # 抽样
        frames = self.transform(frames)  # 转换成tensor
        return frames, label

    def get_frames(self, fp, n_frame):
        '''
        function: get n_frame sample images form video
        :param fp: the video of filepath
        :param n_frame: How many images want to get
        :return: the list of images
        '''
        frames = []
        cap = cv.VideoCapture(fp)  # 读取视频
        if cap.isOpened() == False:
            print('Opening video error!')
            print('The video which can not open is: ', fp)
            return -1
        total_frames = self.frame_counted(fp)  # 获取视频的总帧数
        if total_frames < n_frame:
            print("total_frames of video is too short")
            return -2
        sample_rate = total_frames // n_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            pos = cap.get(cv.CAP_PROP_POS_FRAMES)  # 获取当前帧的位置
            if pos==1 or (pos-1) % sample_rate == 0:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) == n_frame:
                cap.release()
                break
        return np.array(frames)

    def frame_counted(self, fp):
        '''
        function: counting the total frame of video
        :param fp: the filepath of video
        :return: the total frames of video
        '''
        cap = cv.VideoCapture(fp)
        total_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            total_frames = cap.get(cv.CAP_PROP_POS_FRAMES)
        return total_frames

    def transform(self, frames):
        '''
        function: transform image to tensor
        :param frames: image list
        :return: tensor of image
        '''
        frames = [Image.fromarray(frame) for frame in frames]  # 把 numpy 转化成 PIL Image
        transf = transforms.Compose(
            [
                transforms.Resize((320, 256)),  # 缩放成（320, 256）
                transforms.RandomCrop(224),  # 从中随机截取（224, 224）
                transforms.ToTensor(),  # 转换成tensor，每个像素值为[0, 1.0]
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 把取值范围设置成[-1.0, 1.0]
            ]
        )
        frames = [transf(frame) for frame in frames]
        frames = torch.stack(frames).permute(1, 0, 2, 3)  # 把16张tensor堆叠成tensor，并把时序维度和通道维度交换
        return frames

    def __len__(self):
        return len(self.videos)


if __name__ == '__main__':
    fp = "../Datasets/"
    traindata = VideoDataset('train.txt')
    valdata = VideoDataset('val.txt')
    print(len(traindata))
    print(len(valdata))

