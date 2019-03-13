class InvertedResidualNoGroup(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualNoGroup, self).__init__()
        # added by gaowei
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1,  bias=False),
            BN(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            BN(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']


    def forward(self, x):
        t = x
        for name in self.names:
            module = self.conv._modules[name]
            x = module(x)

        if self.use_res_connect:
            return t + x
        else:
            return x



class OptimusPrimeBase(nn.Module):
    def __init__(self, input_dim, label_dim, group_size, group, sync_stats,deconv_setting,
            interverted_residual_setting, input_size=224, width_mult=1., mid_channel = 3):
        super(OptimusPrimeBase, self).__init__()
        #setting of sync BN
        global BN

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=sync_stats, eps=1e-5, momentum=0.05)
        BN = BNFunc


        self.deconv_setting = deconv_setting
        self.interverted_residual_setting = interverted_residual_setting


        # building first layer
        assert input_size % 32 == 0
        self.first_deconv_channel = input_dim
        self.first_conv_channel = 16
##        self.mid_channel = 3
        self.mid_channel = mid_channel
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        input_channel = int(self.first_deconv_channel * width_mult)

        self.features = []
        for c, k, s, p in self.deconv_setting:
            output_channel = int(c * width_mult)
            self.features.append(deconv_bn(input_channel, output_channel, k, s, p))
            input_channel = output_channel

        self.features.append(nn.ConvTranspose2d(input_channel, self.mid_channel, 4, stride=2, padding=1))
        self.features.append(nn.Tanh())
        output_channel = int(self.first_conv_channel * width_mult)
        self.features.append(nn.Conv2d(self.mid_channel, output_channel, 1, stride=1))
        self.features.append(BN(output_channel))
        input_channel = output_channel

        #self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(8, ceil_mode=True))   # such that easily converted to caffemodel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(self.last_channel, label_dim),
        )

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, self.first_deconv_channel, 1, 1)
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class OptimusPrimeBaseNoGroup(nn.Module):
    def __init__(self, input_dim, label_dim, group_size, group, sync_stats,deconv_setting,
            interverted_residual_setting, input_size=224, width_mult=1.):
        super(OptimusPrimeBaseNoGroup, self).__init__()
        #setting of sync BN
        global BN

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=sync_stats, eps=1e-5, momentum=0.05)
        BN = BNFunc


        self.deconv_setting = deconv_setting
        self.interverted_residual_setting = interverted_residual_setting


        # building first layer
        assert input_size % 32 == 0
        self.first_deconv_channel = input_dim
        self.first_conv_channel = 16
        self.mid_channel = 3
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        input_channel = int(self.first_deconv_channel * width_mult)

        self.features = []
        for c, k, s, p in self.deconv_setting:
            output_channel = int(c * width_mult)
            self.features.append(deconv_bn(input_channel, output_channel, k, s, p))
            input_channel = output_channel

        self.features.append(nn.ConvTranspose2d(input_channel, self.mid_channel, 4, stride=2, padding=1))
        self.features.append(nn.Tanh())
        output_channel = int(self.first_conv_channel * width_mult)
        self.features.append(nn.Conv2d(self.mid_channel, output_channel, 1, stride=1))
        self.features.append(BN(output_channel))
        input_channel = output_channel

        #self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidualNoGroup(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidualNoGroup(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(8, ceil_mode=True))   # such that easily converted to caffemodel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(self.last_channel, label_dim),
        )

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, self.first_deconv_channel, 1, 1)
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()




def OptimusPrimeV1(load_pretrain=True, T=6, input_dim = 128, label_dim = 128, group_size=1, group=None, sync_stats=False,input_size=224, width_mult=1.):
    #model = MobileNetV2(T, feature_dim, input_size, width_mult)
    deconv_setting = []
    # setting of deconv blocks
    deconv_setting = [
        #c, k, s, p
        [512, 4, 1, 0],
        [256, 4, 2, 1],
        [128, 4, 2, 1],
        [64, 4, 2, 1],
    ]
    # setting of inverted residual blocks
    interverted_residual_setting = [
        # t, c, n, s
        [T, 32, 3, 2],
        [T, 64, 4, 2],
        [T, 96, 3, 1],
        [T, 160, 3, 2],
        [T, 320, 1, 1],
    ]

    model = OptimusPrimeBase( input_dim, label_dim, group_size, group, sync_stats,deconv_setting,
            interverted_residual_setting )
    return model





def OptimusPrimeV1NoGroup(load_pretrain=True, T=6, input_dim = 128, label_dim = 128, group_size=1, group=None, sync_stats=False,input_size=224, width_mult=1.):
    #model = MobileNetV2(T, feature_dim, input_size, width_mult)
    deconv_setting = []
    # setting of deconv blocks
    deconv_setting = [
        #c, k, s, p
        [512, 4, 1, 0],
        [256, 4, 2, 1],
        [128, 4, 2, 1],
        [64, 4, 2, 1],
    ]
    # setting of inverted residual blocks
    interverted_residual_setting = [
        # t, c, n, s
        [T, 32, 3, 2],
        [T, 64, 4, 2],
        [T, 96, 3, 1],
        [T, 160, 3, 2],
        [T, 320, 1, 1],
    ]

    model = OptimusPrimeBaseNoGroup( input_dim, label_dim, group_size, group, sync_stats,deconv_setting,
            interverted_residual_setting )
    return model



def OptimusPrimeV5NoGroup(load_pretrain=True, T=6, input_dim = 128, label_dim = 128, group_size=1, group=None, sync_stats=False,input_size=224, width_mult=1.):

    # setting of deconv blocks
    deconv_setting = [
        #c, k, s, p
        [128, 4, 1, 0],
        [64, 4, 2, 1],
        [32, 4, 2, 1],
        [16, 4, 2, 1],
    ]

    # setting of inverted residual blocks
    interverted_residual_setting = [
        # t, c, n, s
        [T, 8, 3, 2],
        [T, 16, 4, 2],
        [T, 24, 3, 1],
        [T, 40, 3, 2],
        [T, 80, 1, 1],
    ]

    model = OptimusPrimeBaseNoGroup( input_dim, label_dim, group_size, group, sync_stats,deconv_setting,
            interverted_residual_setting )
    return model



