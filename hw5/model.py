import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, opts):
        super(Embedding, self).__init__()
        self.look_up = nn.Embedding()
        self.dropout = nn.Dropout()
    
    def forward(batch_seq):
        return self.Dropout(self.look_up(batch_seq))



class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.embedding = Embedding()
        self.rnn_cell_l = nn.LSTMCell()
        self.rnn_cell_r = nn.LSTMCell()
        self.dropout = nn.Dropout()

    def forward(self, seq_src):
        seq_emb = self.embedding(seq_src)
        max_len = seq_src.size(0)

        seq_hidden_list_l = []
        seq_hidden_list_r = []
        for i in range(max_len):
            pass
        
        seq_hidden_l = torch.cat()
        seq_hidden_r = torch.cat()

        seq_context = torch.cat()
        return seq_context

class GlobalAttention(nn.Module):
    def __init__(self, opts):
        super(GlobalAttention, self).__init__()
        self.linear = nn.Linear()
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_context, prev_hidden):
        return curr_attn
        
class Decoder(nn.Module):
    def __init__(self, opts):
        super(GlobalAttention, self).__init__()
        self.linear = nn.Linear()
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_context, prev_hidden):
        return curr_attn
    



class NMT(nn.Module):
    def __init__(self, opts):
        super(NMT, self).__init__()
        self.encoder = Encoder(opts)
        self.decoder = Decoder(opts)
    
    def load_param(self,):
        pass
    
    def forward(self, seq_src, seq_trg):
        seq_context = Encoder(seq_src)
        seq_prob = Decoder(seq_context, seq_trg)

        return seq_prob
        
    
