import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro_modules.baselayer import *
from task_utils.utils_pretrain import spclt_copy
from torch.utils.data import Dataset, DataLoader


class custom_encode_dataset(Dataset): 
    def __init__(self, X, T=15):
        self.X = X[:, :-T, :, :]

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        return X, idx
    

class DGCN(nn.Module):
    def __init__(self, para, A, B, return_interpret=False, uncertainty=True, encoder=None):
        super(DGCN, self).__init__()
        self.loader = 'MacroTraffic'

        self.N = para['nb_node']
        self.F = para['dim_feature']
        self.A = A
        self.B = B
        self.T = para['horizon']
        self.interpret = return_interpret
        self.uncertainty = uncertainty
        
        if encoder is None:
            self.encoder = spclt_copy(self.loader).net
        else:
            self.encoder = encoder
        self.decoder = Decoder(self.N, self.F, self.A, self.B, self.T, self.uncertainty)

    def forward(self, x, thred):
        h = self.encoder(x[:,:-self.T])
        prediction, mask1, mask2, demand = self.decoder(x[:,-self.T:], h, thred)

        if self.interpret:
            return prediction, mask1, mask2, demand
        return prediction # (B, n_node, T, dim_feature)
    
    def encode(self, data, batch_size=None, encoding_window=None):
        org_training = self.encoder.training
        self.encoder.eval()
        if batch_size is None:
            batch_size = 16

        if isinstance(data, torch.Tensor):
            dataset = custom_encode_dataset(data, self.T)
        else:
            dataset = custom_encode_dataset(torch.from_numpy(data).float(), self.T)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            output = []
            for x, _ in dataloader:
                x = x[:,:-self.T]
                x = x.to(device)
                out = self.encoder(x) # (B, n_node=193, dim_feature/2=64)
                output.append(out)
            output = torch.cat(output, dim=0)
        
        self.encoder.train(org_training)
        return output


class MSE_scale(nn.Module):
    def __init__(self):
        super(MSE_scale, self).__init__()
        self.main = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.main(inputs*10, targets*10)

    def get_weights(self, label):
        weights = torch.where(label>0.45, 1., 3.)
        return weights
    
    
class LSTM(nn.Module):
    def __init__(self, para, num_layers, encoder=None):
        super(LSTM, self).__init__()
        self.loader = 'MacroLSTM'
        self.T = para['horizon']
        self.input_dim = para['nb_node']*2
        self.hidden_dim = para['nb_node']*4
        self.output_dim = para['nb_node']*2
        self.num_layers = num_layers

        if encoder is None:
            self.encoder = spclt_copy(self.loader).net
        else:
            self.encoder = encoder
        
        self.decoder_list = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        ) for _ in range(self.T)])

    def forward(self, x, thred):
        # Reshape input from (B, seq_len, nb_node, 2) to (B, seq_len, nb_node*2)
        x = x[:,:-self.T]
        x = x.view(x.size(0), x.size(1), -1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Encoder: Process input sequence
        _, (hidden, cell) = self.encoder(x, (h0, c0))

        # Pass the final hidden state through the fully connected layer
        output = torch.stack([decoder(hidden[-1]) for decoder in self.decoder_list], dim=1) # (B, T, output_dim)

        # Reshape output from (B, T, output_dim=193*2) to (B, T, nb_node, 2)
        output = output.view(output.size(0), output.size(1), -1, 2)

        return output # (B, T, nb_node, 2)
    
    def encode(self, data, batch_size=None, encoding_window=None):
        org_training = self.encoder.training
        self.encoder.eval()
        if batch_size is None:
            batch_size = 64

        if isinstance(data, torch.Tensor):
            dataset = custom_encode_dataset(data, self.T)
        else:
            dataset = custom_encode_dataset(torch.from_numpy(data).float(), self.T)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            output = []
            for x, _ in dataloader:
                # Reshape input from (B, seq_len, nb_node, dim_feature) to (B, seq_len, nb_node*dim_feature)
                x = x[:,:-self.T]
                x = x.view(x.size(0), x.size(1), -1).to(device)

                # Initialize hidden and cell states
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                
                # Encoder: Process input sequence
                _, (hidden, cell) = self.encoder(x, (h0, c0))
                out = hidden[-1].view(x.size(0), 193, -1) # (B, nb_node, 4)

                output.append(out)
            output = torch.cat(output, dim=0)
        
        self.encoder.train(org_training)
        return output


class GRU(nn.Module):
    def __init__(self, para, num_layers, encoder=None):
        super(GRU, self).__init__()
        self.loader = 'MacroGRU'
        self.T = para['horizon']
        self.input_dim = para['nb_node']*2
        self.hidden_dim = para['nb_node']*4
        self.output_dim = para['nb_node']*2
        self.num_layers = num_layers

        if encoder is None:
            self.encoder = spclt_copy(self.loader).net
        else:
            self.encoder = encoder

        self.decoder_list = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        ) for _ in range(self.T)])

    def forward(self, x, thred):
        # Reshape input from (B, seq_len, nb_node, 2) to (B, seq_len, nb_node*2)
        x = x[:,:-self.T]
        x = x.view(x.size(0), x.size(1), -1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Encoder: Process input sequence
        _, hidden = self.encoder(x, h0)

        # Pass the final hidden state through the fully connected layer
        output = torch.stack([decoder(hidden[-1]) for decoder in self.decoder_list], dim=1) # (B, T, output_dim)

        # Reshape output from (B, T, output_dim=193*2) to (B, T, nb_node, 2)
        output = output.view(output.size(0), output.size(1), -1, 2)

        return output # (B, T, nb_node, 2)

    
    def encode(self, data, batch_size=None, encoding_window=None):
        org_training = self.encoder.training
        self.encoder.eval()
        if batch_size is None:
            batch_size = 64

        if isinstance(data, torch.Tensor):
            dataset = custom_encode_dataset(data, self.T)
        else:
            dataset = custom_encode_dataset(torch.from_numpy(data).float(), self.T)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)        

        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            output = []
            for x, _ in dataloader:
                # Reshape input from (B, seq_len, nb_node, dim_feature) to (B, seq_len, nb_node*dim_feature)
                x = x[:,:-self.T]
                x = x.view(x.size(0), x.size(1), -1).to(device)

                # Initialize hidden states for the encoder
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

                # Encoder: Process input sequence
                _, hidden = self.encoder(x, h0)
                out = hidden[-1].view(x.size(0), 193, -1) # (B, nb_node, 4)

                output.append(out)
            output = torch.cat(output, dim=0)
        
        self.encoder.train(org_training)
        return output
