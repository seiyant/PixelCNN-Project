import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=0) for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])

    def forward(self, u, ul, embedding=None):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, embedding=embedding)
            ul = self.ul_stream[i](ul, a=u, embedding=embedding)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=2) for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, embedding=None):
        for i in range(self.nr_resnet):
            skip_u = u_list.pop() if len(u_list) > 0 else None #use previous skip connection
            skip_ul = ul_list.pop() if len(ul_list) > 0 else None
            u  = self.u_stream[i](u, a=skip_u, embedding=embedding)
            if skip_ul is not None: #if skip, concatenate with u before ul stream
                ul_in = torch.cat((u, skip_ul), 1)
            else:
                ul_in = u
            ul = self.ul_stream[i](ul, a=ul_in, embedding=embedding)
        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4, embedding_dim=32):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.embedding = nn.Embedding(num_classes, embedding_dim) #embed class labels into 32x vector
        in_channels = input_channels + 1 + embedding_dim #adding embedding channels

        self.u_init = down_shifted_conv2d(in_channels, nr_filters, filter_size=(2,3), shift_output_down=True) #input in_channels

        self.ul_init = nn.ModuleList([down_shifted_conv2d(in_channels, nr_filters, filter_size=(1,3), shift_output_down=True), down_right_shifted_conv2d(in_channels, nr_filters, filter_size=(2,1), shift_output_right=True)]) #input in_channels

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, sample=False, class_label=None):
        embedding = None
        if class_label is not None: #embed and fuse model
            raw_embedding = self.embedding(class_label)
            _, _, H, W = x.shape
            spatial_embedding = raw_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W) #size (B,embedding_dim,H,W)
            x = torch.cat((x, spatial_embedding), dim=1) #size (B,3+embedding_dim,H,W)

        #similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1) 
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], embedding=raw_embedding)
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, embedding=raw_embedding)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)