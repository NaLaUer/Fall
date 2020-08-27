import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)


from config.args import setting_args
from src.util import *

torch.backends.cudnn.benchmark = True

def main():
    parser = setting_args()
    args = parser.parse_args()

    print ("device is available : ", torch.cuda.is_available())
    print ("Start.....")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.dirname(dir_path)
    checkpoint = os.path.join(dir_path, 'save', args.checkpoint)
    model = load_model(args, checkpoint)

    video = os.path.join(dir_path, "input.mp4")

    video_reading (model, video, dir_path)
    print ("Finish")


if __name__ == '__main__':
    main()
