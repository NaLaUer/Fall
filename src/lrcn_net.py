import torch
import  torch.nn as nn
from torchvision import models
import os

class ConvLstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers,
                 bidirectional, n_class, args):
        super(ConvLstm, self).__init__()
        self.conv_model = Conv_Block (latent_dim, args)
        self.Lstm_model = Lstm_Block (latent_dim, hidden_size,
                                      lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(
                2 * hidden_size if bidirectional == True else hidden_size,
                n_class))

    def forward(self, x):

        batch_size, time_steps, channel_x, h_x, w_x = x.shape

        conv_input = x.view(batch_size*time_steps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)

        lstm_input = conv_output.view(batch_size, time_steps, -1)
        lstm_output = self.Lstm_model(lstm_input)

        lstm_output = torch.mean(lstm_output, 1)
        output = self.output_layer(lstm_output)
        return output

class Conv_Block(nn.Module):
    def __init__(self, latent_dim, args):
        super(Conv_Block, self).__init__()

        self.real_path = os.path.realpath(__file__)
        self.dir_path  = os.path.dirname(self.real_path)
        self.sup_path  = os.path.dirname(self.dir_path)

        self.conv_model = models.resnet18(pretrained = args.download_model)
        if args.download_model == False:
            pre = torch.load(os.path.join(self.sup_path,args.pre_model_dir))
            self.conv_model.load_state_dict(pre)

        for param in self.conv_model.parameters():
            param.requires_grad = False

        self.conv_model.fc = nn.Linear(
                self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)

class Lstm_Block (nn.Module):
    def __init__ (self, latent_dim, hidden_size,
                  lstm_layers, bidirectional):
        super(Lstm_Block, self).__init__()
        self.Lstm = nn.LSTM (latent_dim, hidden_size=hidden_size,
                             num_layers=lstm_layers, bidirectional=bidirectional)
        self.hidden_state = None

    def resnet_hidden_state (self):
        self.hidden_state = None

    def forward (self, x):
        output, self.hidden_state = self.Lstm (x, self.hidden_state)
        return output

if __name__ == '__main__':
    inputs = torch.rand(8, 16, 3, 112, 112)

    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # __file__获取执行文件相对路径，整行为取上一级的上一级目录
    sys.path.append(BASE_DIR)
    print (BASE_DIR)
    from config.args import setting_args

    parser = setting_args()
    args = parser.parse_args()


    net = ConvLstm(latent_dim = 128, hidden_size = 128, lstm_layers = 1, bidirectional = False, n_class = 101, args=args)

    outputs = net.forward(inputs)
    print(outputs.size())