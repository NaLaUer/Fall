import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
import torch
import  numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDataset (Dataset):
    def __init__(self, args, split = 'train'):

        self.root_dir = args.root_dir
        self.output_dir = args.out_dir
        self.pic_dir = args.pic_dir
        self.args = args

        folder = os.path.join(self.output_dir, split)

        self.split = split
        self.clip_len = args.clip_len

        self.resize_height = 224
        self.resize_width  = 224

        # 1 判断原始video数据是否存在
        if self.check_integrity() == False:
            raise RuntimeError ("Dataset is empty !")

        # 2 判断切分文件夹是否存在
        if self.check_preprocess() == True:
            print ("数据集创建中，请稍等.....")
            self.preprocess()

        self.fnames = []

        for label in os.listdir(folder):
            fname = os.path.join(folder, label)
            self.fnames.append (fname)

        self.transform = transforms.RandomHorizontalFlip()
        self.ToTensor  = transforms.ToTensor ()

    def __len__ (self):
        return len (self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
    #   buffer = self.crop(buffer, self.clip_len)
        labels = np.array(self.label_array(self.fnames[index]))
        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)


    def label_array (self, fname):
        type = fname.split('/')[-1].split('_')[0]
        if type == 'normal':
            return 1
        else:
            return 0

    def load_frames(self, file_dir):
        frames = sorted(
            [os.path.join(file_dir, img) for img in os.listdir(file_dir)])

        frame_count = len(frames)

        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3),
                          np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len):

        time_index = np.random.randint(buffer.shape[0] - clip_len)
        buffer = buffer[time_index:time_index + clip_len, :,:, :]

        return buffer

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((0, 3, 1, 2))

    def check_integrity(self):
    # function : 判断原始数据集是否存在
        if os.path.exists(self.root_dir):
            return True
        else:
            return False

    def check_preprocess(self):
    # function : 判断切分文件夹是否存在
        if os.path.exists(self.pic_dir):
            return False
        if os.path.exists(self.output_dir):
            return False
        if (os.path.exists(os.path.join(self.root_dir, 'train'))):
            return False
        return True

    def preprocess (self):
        """
        function : 数据集创建， 创建文件夹， 将Video => 图片
        格式 ：
            data
                -train
                    - fall
                    - normal
                -test
                -val
        """
        # 1 创建相关文件夹
        self.mkdir_begin()

        # 2 将所有的图片切分出来
        self.Convert_video()

        # 3 将图片切分到训练集 测试集 验证集
        train, test, val = self.Split_pic()

        # 5 将 Pic 中的数据分别放到训练集、测试集、验证集
        self.Copy_to_split(train, 'train')
        self.Copy_to_split(test, 'test')
        self.Copy_to_split(val, 'val')

        print ("数据创建完毕")

    def Split_pic (self):
    # function : 切分训练集和测试集以及验证集
        file_dir = []
        for out_type_dir in os.listdir(self.pic_dir):
            temp_dir = os.path.join(self.pic_dir, out_type_dir)
            for out_name_dir in os.listdir(temp_dir):
                out_name_dir = os.path.join(temp_dir, out_name_dir)
                for file_name in os.listdir(out_name_dir):
                    file_name = os.path.join(out_name_dir, file_name)
                    file_dir.append(file_name)

        # 4 将数据切分为训练集和测试机以及验证集
        train_and_valid, test = train_test_split(file_dir,
                    test_size=self.args.split_size, random_state=self.args.random)
        train, val = train_test_split(train_and_valid,
                    test_size=self.args.split_size, random_state=self.args.random)

        return train, test, val

    def Convert_video (self):
        # function : 将video 切换为图片
        for file_dir in os.listdir(self.root_dir):
            file_dir = os.path.join(self.root_dir, file_dir)
            # file_dir = /home/sky/PycharmProjects/data/Video/Coffee_room_02
            # 2.1 取出每个图像的摔倒帧数位置，格式为：{"video.txt" : [95, 125]}
            dic = self.Statpos_of_fall(file_dir, 'Annotation_files')
            # 2.2 保存图像，图像名字 label_video_帧数.jpg
            self.Video_to_Pic (file_dir, 'Videos', dic)


    def mkdir_begin (self):
        # function : 创建相关文件夹
        # 0 切割图片文件存储
        if not os.path.exists(self.pic_dir):
            os.mkdir(self.pic_dir)
            os.mkdir(os.path.join(self.pic_dir, 'fall'))
            os.mkdir(os.path.join(self.pic_dir, 'normal'))
        # 1 创建训练集、测试、验证文件
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))
            os.mkdir(os.path.join(self.output_dir, 'val'))


    def Copy_to_split (self, Block, type):
        # function : 将图片从Pic拷贝到split文件夹中
        for src_dir in Block:
            video_name = src_dir.split('/')[-1]
            num = 0
            file_dir = os.listdir(src_dir)
            len_file = len(file_dir)

            for start in range(len_file):
                end = start + self.clip_len
                if end > len_file:
                    break
                num += 1
                dst_dir = os.path.join(self.output_dir,type,video_name+str(num))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                for i in range(start, start + self.clip_len):
                    shutil.copy(os.path.join(src_dir, file_dir[i]), dst_dir)


    def Statpos_of_fall (self, file_dir, video_dir):
        # function : 从 Annotation_files 文件中提取跌倒帧数的起始于终止位置
        video_dir = os.path.join(file_dir, video_dir)
        dic = {}

        for file in os.listdir(video_dir):
            annotation_dir = os.path.join(video_dir, file)
            with open(annotation_dir, 'r') as f:
                line = f.readlines()[:2]
                f.close()
            # 排除 没有帧数的表
            if len(line[0].split(',')) == 1:
                dic[file] = line

        return dic


    def Video_to_Pic (self, file_dir, video_dir, dic):
        """
            function ： 将视频文件切分成图片
            保存形式 ：
                Pic
                    -fall
                        -Coffee_room_02
                            -video (49)
                        -Home_01
                        -Home_02
                    -normal
            注意 ： normal的数量与fall数量是一致的
        """
        video_dir = os.path.join(file_dir, video_dir)
        for video in os.listdir(video_dir):

            if video.split('.')[0]+'.txt' in dic:
                line = dic[video.split('.')[0]+'.txt']

                video_path = os.path.join(video_dir, video)

                index_list = [int(val.strip()) for val in line]

                self.Read_Video(video_path, video, index_list)

    def Read_Video (self, video_path, video, index_list):
        #function ： 读取视频文件
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = 0

        video_path_dir = video_path.split('/')[-3]
        video_name = video.split('.')[0]
        while (count < frame_count):
            retaining, frame = capture.read()

            if frame is None:
                continue

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            if count < index_list[1] and count > index_list[0]:
                self.Save_pic(self.pic_dir, 'fall', frame, count, video_path_dir, video_name)

            elif count < index_list[0] and count > (index_list[0])*2 - index_list[1]:
                self.Save_pic(self.pic_dir, 'normal', frame, count, video_path_dir, video_name)
            count += 1


    def Save_pic (self, pic_dir, type, frame, count, video_path_dir, video_name):
        """
            function : 保存文件
            /home/sky/PycharmProjects/data/Pic/fall/Coffee_room_02/video (49)/000345.jpg
        """
        input_dir = os.path.join(pic_dir, type, video_path_dir)
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)
        input_dir = os.path.join(input_dir, type + '_' + video_path_dir + '_' + video_name)
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)
        cv2.imwrite(filename = os.path.join(input_dir,'0000{}.jpg'.format(str(count))),
                    img = frame)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
    sys.path.append(BASE_DIR)
    print (BASE_DIR)
    from config.args import setting_args

    parser = setting_args()
    args = parser.parse_args()

    train_data = VideoDataset(args,split='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break