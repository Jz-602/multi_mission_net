import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self,
                 n_filters=[64, 128],
                 filter_sizes=[1],
                 strides=[1],
                 b_norm=False,
                 verbose=False,
                 bneck_size=128,
                 input_size=2048):
        super(Encoder, self).__init__()
        if verbose:
            print('Building Encoder')
        n_filters.append(bneck_size)
        n_layers = len(n_filters)
        self.input_size = input_size
        self.bneck_size = bneck_size
        self.conv = []
        self.bn = []
        for i in range(n_layers):
            if i == 0:
                in_channels = 3
            else:
                in_channels = n_filters[i - 1]
            self.conv.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=n_filters[i],
                    kernel_size=filter_sizes[0],
                    stride=strides[0]))
            if (b_norm):
                self.bn.append(
                    nn.BatchNorm1d(
                        num_features=in_channels,
                        eps=1e-05,
                        momentum=None,
                        affine=True))
        for i, layer in enumerate(self.conv, 1):
            setattr(self, 'conv_{}'.format(i), layer)
        for i, layer in enumerate(self.bn, 1):
            setattr(self, 'bn_{}'.format(i), layer)

    def forward(self,
                x,
                non_linearity=F.relu,
                regularizer=None,
                weight_decay=0.001,
                dropout_prob=None,
                pool=F.avg_pool1d,
                pool_sizes=None,
                padding='same',
                verbose=False,
                closing=None):
        n_layers = len(self.conv)
        b_norm = len(self.bn)
        for i in range(n_layers):
            x = self.conv[i](x)
            if b_norm:
                x = self.bn[i](x)

            if non_linearity is not None:
                x = non_linearity(x)

            if pool is not None and pool_sizes is not None:
                if pool_sizes[i] is not None:
                    x = pool(x, kernel_size=pool_sizes[i])
            if dropout_prob is not None and dropout_prob[i] > 0:
                x = F.dropout(x, 1.0 - dropout_prob[i])
            if verbose:
                print(x.size())
        x, _ = torch.max(x, dim=2)
        if verbose:
            print(x.size())

        if closing is not None:
            x = closing(x)
            if verbose:
                print(x)
                
        return x


class Decoder(nn.Module):

    def __init__(self,
                 layer_sizes=[1024, 2048 * 3],
                 b_norm=False,
                 non_linearity=F.relu,
                 regularizer=None,
                 weight_decay=0.001,
                 b_norm_finish=False,
                 verbose=False,
                 bneck_size=128):

        super(Decoder, self).__init__()
        if verbose:
            print('Building Decoder')
        self.bneck_size = bneck_size
        layer_sizes.insert(0, bneck_size)
        n_layers = len(layer_sizes)
        self.out_size = int(layer_sizes[n_layers - 1] / 3)
        self.fc = []
        self.bn = []
        self.bn_finish = None
        for i in range(n_layers - 1):
            self.fc.append(
                nn.Linear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1]))
            if (b_norm and i < n_layers - 1):
                self.bn.append(
                    nn.BatchNorm1d(
                        num_features=layer_sizes[i],
                        eps=1e-05,
                        momentum=None,
                        affine=True))
            if (b_norm_finish and i == n_layers - 1):
                self.bn_finish = nn.BatchNorm1d(
                    num_features=layer_sizes[i],
                    eps=1e-05,
                    momentum=None,
                    affine=True)
        for i, layer in enumerate(self.fc, 1):
            setattr(self, 'fc_{}'.format(i), layer)
        for i, layer in enumerate(self.bn, 1):
            setattr(self, 'bn_{}'.format(i), layer)

    def forward(self,
                x,
                non_linearity=F.relu,
                regularizer=None,
                weight_decay=0.001,
                dropout_prob=None,
                verbose=False):
        n_layers = len(self.fc)
        b_norm = len(self.bn)
        for i in range(n_layers - 1):
            x = self.fc[i](x)
            if b_norm:
                x = self.bn[i](x)

            if non_linearity is not None:
                x = non_linearity(x)

            if dropout_prob is not None and dropout_prob[i] > 0:
                x = F.dropout(x, 1.0 - dropout_prob[i])
            if verbose:
                print(x.size())
        x = self.fc[n_layers - 1](x)
        if (self.bn_finish is not None):
            x = self.bn_finish(x)
        if verbose:
            print(x.size())
        x = x.view(x.shape[0], 3, self.out_size)
        return x




class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, chkpt_file=None):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bneck_size = encoder.bneck_size
        self.load_chkpt(chkpt_file=chkpt_file)

    def load_chkpt(self, chkpt_file=None):
        if(chkpt_file is not None and chkpt_file is not ""):
            print("=> loading checkpoint '{}'".format(chkpt_file))
            checkpoint = torch.load(chkpt_file)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt_file, checkpoint['epoch']))

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)


class PCAutoEncoder(AutoEncoder):

    def __init__(self, bneck_size=128, chkpt_file=None):
        self.input_size = 2048
        encoder = Encoder(bneck_size=bneck_size, input_size=self.input_size)
        decoder = Decoder(bneck_size=bneck_size)
        super().__init__(encoder, decoder, chkpt_file=chkpt_file)




class Model(nn.Module):

    def __init__(self,
                 bneck_size=128,
                 chkpt_file=None,
                 AE_chkpt_file=None):
        super(Model, self).__init__()
        self.AE = PCAutoEncoder(bneck_size, AE_chkpt_file)
        self.input_size = self.AE.input_size

        self.bneck_size = bneck_size
        self.score = nn.Linear(bneck_size * 2, 1)

        if chkpt_file is not None:
            checkpoint = torch.load(chkpt_file)
            self.load_state_dict(checkpoint['state_dict'])


    def forward(self, this_PC, prev_PC):
        X = self.AE.encode(this_PC)
        Y = self.AE.encode(prev_PC)
        Y_AE = self.AE.forward(prev_PC)
        Sim = F.cosine_similarity(X, Y, dim=1)
        # X = self.score(torch.cat((X,Y),dim=1)).squeeze()
        return Sim, Y_AE
        # return Sim

    def FastForward(self, this_PC, model_code):
        X = self.AE.encode(this_PC)
        Sim = F.cosine_similarity(X, model_code, dim=1)
        return Sim

    def encode(self, X):
        return self.AE.encode(X)

    def decode(self, X):
        return self.AE.decode(X)
