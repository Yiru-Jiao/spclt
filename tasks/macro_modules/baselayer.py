import torch
import torch.nn.functional as F
from torch import nn
import random 

class LocalGC(nn.Module):
    def __init__(self, nb_node, dim_feature, A):
        super(LocalGC, self).__init__()
        self.N = nb_node
        self.F = dim_feature

        self.w1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.N, self.N)))
        self.bias = nn.Parameter(torch.zeros(self.F,))

        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, x):
        W = self.A*self.w1
        x = torch.matmul(x.transpose(-1,-2), W)
        output = x.transpose(-1,-2) + self.bias

        return output
    
class DoubleDGC(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B):
        super(DoubleDGC, self).__init__()
        self.N = nb_node
        self.F = dim_feature

        self.w1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(2, self.F, self.N)))
        self.w2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(2, self.N, self.N)))
        self.bias = nn.Parameter(torch.zeros(2, self.N, 1))

        self.convert = nn.Linear(self.F, 2)

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, x, state):

        demand = self.convert(x)
        demand = torch.tanh(demand)*0.5

        x1 = torch.matmul(x.transpose(-1, -2), self.A*self.w2[0])
        mask1 = torch.matmul(x1.transpose(-1, -2), self.w1[0]) + self.bias[0]
        mask1 = mask1 + -10e15 * (1.0 - self.A)
        mask1 = torch.softmax(mask1, -1)

        x2 = torch.matmul(x.transpose(-1, -2), self.B*self.w2[1])
        mask2 = torch.matmul(x2.transpose(-1, -2), self.w1[1]) + self.bias[1]
        mask2 = mask2 + -10e15 * (1.0 - self.B)
        mask2 = torch.softmax(mask2, -1)

        v = torch.matmul(mask1, state[...,:1])
        q = torch.matmul(mask2, state[...,1:])

        output = torch.cat((v,q), -1) + demand

        return output, mask1, mask2, demand[...,-1]
    
class DGCNcell(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, return_interpret=False):
        super(DGCNcell, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.dim = self.F//2+2

        self.interpret = return_interpret

        self.dgc_r = LocalGC(self.N, self.dim, self.A)
        self.lin_r = nn.Linear(self.dim, self.F//2)

        self.dgc_u = LocalGC(self.N, self.dim, self.A)
        self.lin_u = nn.Linear(self.dim, self.F//2)

        self.dgc_c = LocalGC(self.N, self.F, self.A)
        self.lin_c = nn.Linear(self.F, self.F//2)

        #self.core = DynamicGC(self.N, self.F, self.A)
        self.core = DoubleDGC(self.N, self.F, self.A, self.B)

        #self.lin_out = nn.Linear(self.F, 1)

        self.lin_in = nn.Linear(2, self.F//2)


    def forward(self, input, state):
        #print(state.size())
        x = self.lin_in(input)
        gru_input = torch.cat([x, state], -1)

        p, mask1, mask2, demand = self.core(gru_input, input)
        feature_ru = torch.cat([p, state], -1)

        r = self.dgc_r(feature_ru)
        r = self.lin_r(r)
        r = torch.sigmoid(r)

        u = self.dgc_u(feature_ru)
        u = self.lin_u(u)
        u = torch.sigmoid(u)

        s = r*state
        feature_c = torch.cat([x, s], -1)
        c = self.dgc_c(feature_c)
        c = self.lin_c(c)
        c = torch.tanh(c)

        H = u*state + (1-u)*c
        #print(H.size())
        if self.interpret:
            return  p, H, mask1, mask2, demand
        return p, H  
    
class DGCNcellUQ(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, return_interpret=False, uncertainty=False):
        super(DGCNcellUQ, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.dim = self.F//2+2
        self.uncertainty = uncertainty

        self.interpret = return_interpret

        self.dgc_r = LocalGC(self.N, self.dim, self.A)
        self.lin_r = nn.Linear(self.dim, self.F//2)

        self.dgc_u = LocalGC(self.N, self.dim, self.A)
        self.lin_u = nn.Linear(self.dim, self.F//2)

        self.dgc_c = LocalGC(self.N, self.F, self.A)
        self.lin_c = nn.Linear(self.F, self.F//2)

        #self.core = DynamicGC(self.N, self.F, self.A)
        self.core = DoubleDGC(self.N, self.F, self.A, self.B)

        #self.lin_out = nn.Linear(self.F, 1)

        self.lin_in = nn.Linear(2, self.F//2)
        if self.uncertainty:
            self.lin_reg = nn.Linear(self.F//2, 4)


    def forward(self, input, state):

        x = self.lin_in(input)
        gru_input = torch.cat([x, state], -1)

        p, mask1, mask2, demand = self.core(gru_input, input)
        feature_ru = torch.cat([p, state], -1)

        r = self.dgc_r(feature_ru)
        r = self.lin_r(r)
        r = torch.sigmoid(r)
        
        u = self.dgc_u(feature_ru)
        u = self.lin_u(u)
        u = torch.sigmoid(u)

        s = r*state
        feature_c = torch.cat([x, s], -1)

        c = self.dgc_c(feature_c)
        c = self.lin_c(c)
        c = torch.tanh(c)

        H = u*state + (1-u)*c

        if self.uncertainty:
            Hr = self.lin_reg(H)

            mu = torch.sigmoid(Hr[...,3:4])
            a = F.softplus(Hr[...,:1]) + 2.
            b = F.softplus(Hr[...,1:2])
            v = F.softplus(Hr[...,2:3]+2)

            uncertainty = torch.cat((mu, v, a, b), -1)


            if self.interpret:
                return  p, H, uncertainty, mask1, mask2, demand
            return p, H, uncertainty
        
        if self.interpret:
            return  p, H, mask1, mask2, demand
        return p, H
    
class DGCNcellUQr(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, return_interpret=False, uncertainty=False):
        super(DGCNcellUQr, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.dim = self.F//2+3
        self.uncertainty = uncertainty

        self.interpret = return_interpret

        self.dgc_r = LocalGC(self.N, self.dim, self.A)
        self.lin_r = nn.Linear(self.dim, self.F//2)

        self.dgc_u = LocalGC(self.N, self.dim, self.A)
        self.lin_u = nn.Linear(self.dim, self.F//2)

        self.dgc_c = LocalGC(self.N, self.F, self.A)
        self.lin_c = nn.Linear(self.F, self.F//2)

        self.core = DoubleDGC(self.N, self.F, self.A, self.B)

        self.lin_in = nn.Linear(3, self.F//2)
        if self.uncertainty:
            self.lin_reg = nn.Linear(self.F//2, 4)


    def forward(self, input, state):

        x = self.lin_in(input)
        gru_input = torch.cat([x, state], -1)

        p, mask1, mask2, demand = self.core(gru_input, input[...,:2])
        feature_ru = torch.cat([input, state], -1)

        r = self.dgc_r(feature_ru)
        r = self.lin_r(r)
        r = torch.sigmoid(r)
        
        u = self.dgc_u(feature_ru)
        u = self.lin_u(u)
        u = torch.sigmoid(u)

        s = r*state
        feature_c = torch.cat([x, s], -1)

        c = self.dgc_c(feature_c)
        c = self.lin_c(c)
        c = torch.tanh(c)

        H = u*state + (1-u)*c

        if self.uncertainty:
            Hr = self.lin_reg(H)

            mu = torch.sigmoid(Hr[...,3:4])
            a = F.softplus(Hr[...,:1]) + 2.
            b = F.softplus(Hr[...,1:2])
            v = F.softplus(Hr[...,2:3]+2)

            uncertainty = torch.cat((mu, v, a, b), -1)

            sigma_total = torch.sqrt(b*(v+1)/v/(a-1))

            p = torch.cat((p, sigma_total), -1)

            if self.interpret:
                return  p, H, uncertainty, mask1, mask2, demand
            return p, H, uncertainty
        
        if self.interpret:
            return  p, H, mask1, mask2, demand
        return p, H

class Encoder(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B):
        super(Encoder, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B

        self.encodercell = DGCNcell(self.N, self.F, self.A, self.B)

        self.init_state = nn.Parameter(torch.zeros(self.N, self.F//2), requires_grad=False)
      
    def forward(self, x):
        for i in range(x.size(1)):
            if i==0:
                _, state = self.encodercell(x[:,i], self.init_state.repeat(x.size(0), 1, 1))
            else:
                _, state = self.encodercell(x[:,i], state)

        return state
    
class Decoder(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, nb_step, uncertainty=False):
        super(Decoder, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.T = nb_step
        self.uncertainty = uncertainty
        if self.uncertainty:
            self.decodercell = DGCNcellUQr(self.N, self.F, self.A, self.B, True, self.uncertainty)
        else:
            self.decodercell = DGCNcellUQ(self.N, self.F, self.A, self.B, True, self.uncertainty)
      
    def forward(self, x, init_state, threshold):
        if self.uncertainty:
            tail = torch.ones_like(x[:,0,:,:1])*1e-2
            xin = torch.cat((x[:,0], tail), -1)
            prediction, state, uncertainty, mask1, mask2, demand = self.decodercell(xin, init_state)
            p = [prediction]
            u = [uncertainty]

            for i in range(1, self.T):
                coin = random.uniform(0, 1)
                if coin > threshold:
                    prediction, state, uncertainty, _, _, _ = self.decodercell(prediction, state)
                else:
                    xin = torch.cat((x[:,i], tail), -1)
                    prediction, state, uncertainty,_, _, _ = self.decodercell(xin, state)
                p.append(prediction)
                u.append(uncertainty)
            p = torch.stack(p, 1)
            u = torch.stack(u, 1)

            p = torch.cat((u, p), -1)

            return p, mask1, mask2, demand

        else:
            prediction, state, mask1, mask2, demand = self.decodercell(x[:,0], init_state)
            p = [prediction]
            D = [demand]
            M1 = [mask1]
            M2 = [mask2]

            for i in range(1, self.T):
                coin = random.uniform(0, 1)
                if coin > threshold:
                    prediction, state, mask1, mask2, demand = self.decodercell(prediction, state)
                else:
                    prediction, state, mask1, mask2, demand = self.decodercell(x[:,i], state)
                p.append(prediction)
                D.append(demand)
            p = torch.stack(p, 1)
            D = torch.stack(D, 1)
            M1 = torch.stack(M1, 1)
            M2 = torch.stack(M2, 2)

            return p, M1, M2, D
        
class Decoderi(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, nb_step, uncertainty=False):
        super(Decoderi, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.T = nb_step
        self.uncertainty = uncertainty

        self.decodercell = DGCNcellUQ(self.N, self.F, self.A, self.B, True, self.uncertainty)
      
    def forward(self, x, init_state, threshold):
        if self.uncertainty:
            prediction, state, uncertainty, mask1, mask2, demand = self.decodercell(x[:,0], init_state)
            p = [prediction]
            u = [uncertainty]

            for i in range(1, self.T):
                coin = random.uniform(0, 1)
                if coin > threshold:
                    prediction, state, uncertainty, _, _, _ = self.decodercell(prediction, state)
                else:
                    prediction, state, uncertainty,_, _, _ = self.decodercell(x[:,i], state)
                p.append(prediction)
                u.append(uncertainty)
            p = torch.stack(p, 1)
            u = torch.stack(u, 1)

            p = torch.cat((u, p), -1)

            return p, mask1, mask2, demand

        else:
            prediction, state, mask1, mask2, demand = self.decodercell(x[:,0], init_state)
            p = [prediction]
            D = [demand]
            M1 = [mask1]
            M2 = [mask2]

            for i in range(1, self.T):
                coin = random.uniform(0, 1)
                if coin > threshold:
                    prediction, state, mask1, mask2, demand = self.decodercell(prediction, state)
                else:
                    prediction, state, mask1, mask2, demand = self.decodercell(x[:,i], state)
                p.append(prediction)
                D.append(demand)
            p = torch.stack(p, 1)
            D = torch.stack(D, 1)
            M1 = torch.stack(M1, 1)
            M2 = torch.stack(M2, 2)

            return p, M1, M2, D
        

class Encoder_vis(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B):
        super(Encoder_vis, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B

        self.encodercell = DGCNcell(self.N, self.F, self.A, self.B)

        self.init_state = nn.Parameter(torch.zeros(self.N, self.F//2), requires_grad=False)
      
    def forward(self, x):
        for i in range(x.size(1)):
            if i==0:
                _, state = self.encodercell(x[:,i], self.init_state.repeat(x.size(0), 1, 1))
            else:
                _, state = self.encodercell(x[:,i], state)

        return state
    
class Decoder_vis(nn.Module):
    def __init__(self, nb_node, dim_feature, A, B, nb_step):
        super(Decoder_vis, self).__init__()
        self.N = nb_node
        self.F = dim_feature
        self.A = A
        self.B = B
        self.T = nb_step

        self.decodercell = DGCNcellUQr(self.N, self.F, self.A, self.B, return_interpret=True, uncertainty=True)
      
    def forward(self, x, init_state):
        tail = torch.ones_like(x[:,0,:,:1])*1e-2
        xin = torch.cat((x[:,0], tail), -1)

        # tail = torch.ones_like(x[...,:1])*1e-2
        # xin = torch.cat((x, tail), -1)

        prediction, state, uncertainty, mask1, mask2, demand = self.decodercell(xin, init_state)
        p = [prediction]
        u = [uncertainty]
        M1 = [mask1]
        M2 = [mask2]
        D = [demand]

        for i in range(1, self.T):
            prediction, state, uncertainty, mask1, mask2, d = self.decodercell(prediction, state)
            p.append(prediction)
            u.append(uncertainty)
            M1.append(mask1)
            M2.append(mask2)
            D.append(d)

        p = torch.stack(p, 1)
        u = torch.stack(u, 1)
        M1 = torch.stack(M1, 0).squeeze()
        M2 = torch.stack(M2, 0).squeeze()
        D = torch.stack(D, 1).squeeze()


        p = torch.cat((u, p), -1).squeeze()

        return p, M1, M2, D

