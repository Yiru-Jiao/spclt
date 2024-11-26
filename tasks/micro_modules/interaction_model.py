import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from micro_modules.baselayers import *
from micro_modules.encoder import *
from micro_modules.decoder import *
from torch.utils.data import Dataset, DataLoader


class custom_encode_dataset(Dataset): 
    def __init__(self, X):
        self.X = X

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        X = self.X[idx]
        return X, idx
    

class UQnet(nn.Module):
    def __init__(self, para, test=False, drivable=True, traj_encoder=None):
        super(UQnet, self).__init__()
        self.loader = 'MicroTraffic'

        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.resolution = para['resolution']
        self.test = test
        self.prob_mode = para['prob_mode']
        self.inference = para['inference']
        
        if traj_encoder is None:
            self.encoder = VectorEncoder(para)
        else:
            self.encoder = VectorEncoder(para, traj_encoder)
        decoder_dims = para['encoder_attention_size']
        
        lateral = torch.tensor([i+0.5 for i in range(int(-self.xmax/self.resolution), 
                                                         int(self.xmax/self.resolution))])*self.resolution
        longitudinal = torch.tensor([i+0.5 for i in range(int(self.ymin/self.resolution), 
                                                     int(self.ymax/self.resolution))])*self.resolution

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        x1 = lateral.repeat(self.len_y, 1).transpose(1,0)
        y1 = longitudinal.repeat(self.len_x, 1)
        self.mesh = nn.Parameter(torch.stack((x1,y1),-1),requires_grad = False)

        self.decoder = VectorDecoder(para, drivable)
        if not self.inference:
            self.reg_decoder = RegularizeDecoder(para, drivable)
       
    def forward(self, trajectory, maps, masker, lanefeatures, adj, af, ar, c_mask):
        if self.inference:
            hlane, hmid, hinteraction = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        else:
            hlane, hmid, hinteraction, hmae = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        grid = self.mesh.reshape(-1, 2)
        log_lanescore, heatmap = self.decoder(hlane, hmid, hinteraction, grid, c_mask, masker)
        heatmap = heatmap.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
        
        if not self.inference:
            heatmap_reg = self.reg_decoder(hmae, grid, ar, c_mask, masker)
            heatmap_reg = heatmap_reg.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
            return log_lanescore, heatmap, heatmap_reg
        else:
            if not self.test:
                return log_lanescore, heatmap
            else:
                if self.prob_mode=='nll':
                    out = torch.exp(heatmap)#*masker
                else:
                    out = torch.sigmoid(heatmap)
                    out = torch.clamp(out, min=1e-7)
                    if self.resolution==0.5:
                        out = nn.AvgPool2d(3,stride=1,padding=1)(out.unsqueeze(0))
                return torch.exp(log_lanescore), out.squeeze()
            
    def encode(self, data, batch_size=None, encoding_window=None):
        org_training = self.encoder.traj_encoder.training
        self.encoder.traj_encoder.eval()
        if batch_size is None:
            batch_size = 8

        if isinstance(data, torch.Tensor):
            dataset = custom_encode_dataset(data)
        else:
            dataset = custom_encode_dataset(torch.from_numpy(data).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        device = next(self.encoder.traj_encoder.parameters()).device
        with torch.no_grad():
            output = []
            for x, _ in dataloader:
                x = x.to(device)
                out = self.encoder.traj_encoder(x)
                output.append(out)
            output = torch.cat(output, dim=0) # (B, n_agents=26, dim_feature=128)
        
        self.encoder.traj_encoder.train(org_training)
        return output


class TrajModel(nn.Module):
    def __init__(self, ):
        super(TrajModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 58)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, encoder=None):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if encoder is None:
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.encoder = encoder
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        _, (hidden, cell) = self.encoder(x, (h0, c0))

        # Use the hidden state to initialize the input for the decoder
        seq_len = 30
        decoder_initial = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1) # (B, seq_len, hidden_dim)
        
        # Forward propagate through LSTM
        output, _ = self.lstm(decoder_initial, (hidden, cell))
        
        # Pass the final hidden state through the fully connected layer
        output = self.fc(output[:, -1, :])  # Use only the last timestep output

        return output # (B, output_dim)
    

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, encoder=None):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if encoder is None:
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.encoder = encoder
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden states for the encoder
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Encoder: Process input sequence
        _, hidden = self.encoder(x, h0)

        # Use the hidden state to initialize the input for the decoder
        seq_len = 30
        decoder_initial = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1) # (B, seq_len, hidden_dim)

        # Decoder: Process the entire sequence at once
        output, _ = self.gru(decoder_initial, hidden.unsqueeze(0))

        # Apply fully connected layer to each time step
        output = self.fc(output[:, -1, :]) # Use only the last timestep output

        return output # (B, output_dim)