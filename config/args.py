import argparse

def setting_args():
    parser = argparse.ArgumentParser(
        description='Fall Detection, LRCN architecture')
    parser.add_argument('--epochs', default=100, type=int,
        help='number of total epochs')
    parser.add_argument('--batch_size', default=16, type=int,
        help='min-batch size')
    parser.add_argument('--lr', default=5e-3, type=float,
        help='initial learning rate')
    parser.add_argument('--num_workers', default=4, type=int,
        help='initial num_workers, the number of processes that'\
             'generate batches in parallel')
    parser.add_argument('--split_size', default=0.2, type=float,
        help='set the size of the split between validation and train')
    parser.add_argument('--clip_len', default=4, type=int,
        help='seq len of video frames')
    parser.add_argument('--random', default=42, type=int,
        help='random of system')
    parser.add_argument('--dataset', default='Le2i',
                        type=str, help='The name of dataset')

    parser.add_argument('--root_dir',default='/hdd01/liuchang/Le2i/Video',
        type=str, help='The dir for orig video dir')
    parser.add_argument('--pic_dir', default='/hdd01/liuchang/Le2i/Pic',
        type=str, help='The pic of video frames')
    parser.add_argument('--out_dir',default='/hdd01/liuchang/Le2i/Video_split',
        type=str, help='split dataset')
    parser.add_argument('--save_dir', default='save', type=str,
        help='save model of training')

    parser.add_argument('--download_model', default=False, type=bool,
        help='download resnet mode, defalut = False')
    parser.add_argument('--pre_model_dir', default='models/resnet18-5c106cde.pth',
        type=str, help='pre-training model dir, if --download_model is True, ignore it')
    
    parser.add_argument('--checkpoint', default='Le2i_epoch-20.pth.tar', type=str,
        help='checkpoint of model')
    parser.add_argument('--input', default='/input.mp4', type=str, help='test video')

    parser.add_argument('--latent_dim', default=128, type=int,
        help='The dim of Conv Fc output')
    parser.add_argument('--hidden_size', default=256, type=int,
        help='The number of features in the LSTM hidden state')
    parser.add_argument('--lstm_layers', default=1, type=int,
        help='The number of layers')
    parser.add_argument('--bidirectional', default=True, type=bool,
        help='set the LSTM to be bidirectional')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='set the classes of result')

    parser.add_argument('--useTest', default=True, type=bool,
        help='see evolution of the test set when training')
    parser.add_argument('--nTestInterval', default=20, type=int,
                        help='Run on test set every nTestInterval epochs')
    parser.add_argument('--snapshot', default=5, type=int,
            help='save net every snapshot epochs')

    return  parser

if __name__ == "__main__":
    parser = setting_args()
    args = parser.parse_args()

    print(args.lr)