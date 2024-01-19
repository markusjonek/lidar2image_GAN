import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()

        self.up_seq = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.down_seq = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 9)
        )
        
    def forward(self, x):
        x_t = self.up_seq(x)
        x_t = self.down_seq(x_t)
        x_t = x_t.view(-1, 3, 3)
        
        x = x.unsqueeze(1)
        x = torch.bmm(x, x_t)
        x = x.squeeze(1)
        return x


class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()

        self.up_seq = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.down_seq = nn.Sequential(
            nn.Linear(512, 32*32)
        )
        
    def forward(self, x):
        x_t = self.up_seq(x)
        x_t = self.down_seq(x_t)
        x_t = x_t.view(-1, 32, 32)
        x = x.unsqueeze(1)
        x = torch.bmm(x, x_t)
        x = x.squeeze(1)
        return x


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.tnet = TNet()
        #self.fnet = FNet()

        self.seq1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32),
        )

        self.seq2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.seq_128_1024 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.attention_seq = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        self.refl_seq = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.seq3 = nn.Sequential(
            nn.Linear(1024 + 128 + 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 3)
        )

    def forward(self, x, batch):
        xyz = x[:, :3]
        refl = x[:, 3].unsqueeze(1)

        refl = self.refl_seq(refl)
        
        x1 = self.tnet(xyz)
        x2 = self.seq1(x1)
        #x3 = self.fnet(x2)
        x4 = self.seq2(x2)

        x_1024 = self.seq_128_1024(x4)

        attention_weights = self.attention_seq(x_1024)
        attention_weights = torch.softmax(attention_weights, dim=0)

        gf = torch.mul(x_1024, attention_weights)

        gf_x = torch.cat([gf[batch], x4, refl], dim=1)
        x5 = self.seq3(gf_x)

        return torch.sigmoid(x5)




class PointNetSMALL(nn.Module):
    def __init__(self):
        super(PointNetSMALL, self).__init__()

        self.seq_3_64 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.seq_64_128 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.seq_128_128 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.seq_128_1024 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # self.refl_seq = nn.Sequential(
        #     nn.Linear(1, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),

        #     nn.Linear(16, 16),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )

        self.seq3 = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )


    def forward(self, x, batch):
        xyz = x[:, :3]
        refl = x[:, 3].unsqueeze(1)

        # refl = self.refl_seq(refl)

        x64 = self.seq_3_64(xyz)
        x128 = self.seq_64_128(x64)
        x128 = self.seq_128_128(x128)
        x1024 = self.seq_128_1024(x128)

        gf = global_max_pool(x1024, batch)
        
        gf_x = torch.cat([gf[batch], x128], dim=1)

        xColor = torch.sigmoid(self.seq3(gf_x))

        return xColor






class PointColorer(nn.Module):
    def __init__(self):
        super(PointColorer, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024)
        )

        self.attention_seq = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        self.seq3 = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, x, batch):
        x1 = self.seq1(x)
        x2 = self.seq2(x1)

        attention_weights = self.attention_seq(x2)
        attention_weights = torch.softmax(attention_weights, dim=0)
        gf = torch.mul(x2, attention_weights)

        x3 = torch.cat([gf[batch], x1], dim=1)
        x4 = self.seq3(x3)
        return torch.sigmoid(x4)



class PointColorerSMALL(nn.Module):
    def __init__(self):
        super(PointColorerSMALL, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1024)
        )

        self.attention_seq = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

        self.seq3 = nn.Sequential(
            nn.Linear(1024 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward(self, x, batch):
        x1 = self.seq1(x)
        x2 = self.seq2(x1)

        attention_weights = self.attention_seq(x2)
        attention_weights = torch.softmax(attention_weights, dim=0)
        gf = torch.mul(x2, attention_weights)

        x3 = torch.cat([gf[batch], x1], dim=1)
        x4 = self.seq3(x3)
        return torch.sigmoid(x4)
