import torch
import torch.nn as nn
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self,
        voc_size,        
        dim_emb,
        dropout=0.5
        ):
        super(Embedding, self).__init__()
        self.look_up = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=dim_emb
        )
        self.dropout = nn.Dropout(
            p=dropout
        )
    
    def forward(self, batch_seq):
        return self.dropout(self.look_up(batch_seq))



class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()

        self.embedding = Embedding(
            opts['src_voc_size'],
            opts['dim_emb'],
            opts['dropout']
        )
        
        
        self.bi_lstm = nn.LSTM(
            input_size=opts['dim_emb'],
            hidden_size=opts['dim_rnn'],
            dropout=opts['dropout'],
            bidirectional=True
        )
        #self.dropout = nn.Dropout(
        #    p=opts['dropout']
        #)

    def forward(self, src_batch, src_mask):
        seq_emb = self.embedding(src_batch)       
        seq_context, final_states = self.bi_lstm(seq_emb)
        #print(seq_context.size())
        return seq_context, final_states


class GlobalAttention(nn.Module):
    def __init__(self, opts):
        super(GlobalAttention, self).__init__()
        self.dim_rnn = opts['dim_rnn']

        self.linear_out = nn.Linear(4 * self.dim_rnn, 2 * self.dim_rnn, bias=False)

        self.dropout = nn.Dropout(p=opts['dropout'])

    def forward(
        self,
        seq_context, 
        seq_context_after_liner, 
        src_mask, 
        prev_hidden
    ):
        # seq_score shape [seq_len, batch_size]
        #import pdb; pdb.set_trace()
        
        #print(seq_context_after_liner.size())
        #print(prev_hidden.size())
        #import pdb; pdb.set_trace()
        seq_score = torch.exp(
            torch.sum(
                seq_context_after_liner * prev_hidden[None, :, :],
                dim=2
            )
        )
        #print(seq_score.size())
        seq_score = seq_score * src_mask.float()

        
        seq_score_sum = torch.sum(seq_score, dim=0, keepdim=True)
        seq_score_sum[seq_score_sum == 0] = 1

        seq_score_norm = seq_score / seq_score_sum

        # seq_context shape [seq_len, batch_size, dim_model * 2]
        #print(seq_score_norm.unsqueeze(2).size())
        #print(seq_context.size())
        weighted_sum_over_src = torch.sum(
            seq_score_norm.unsqueeze(2) * seq_context,
            dim=0
        )

        #print(weighted_sum_over_src.size())
        #print(prev_hidden[0].size())
        
        attn = torch.tanh(
            self.linear_out(
                torch.cat(
                    [
                        weighted_sum_over_src,
                        prev_hidden
                    ],
                    dim=1
                )
            )
        )
        return self.dropout(attn)

class Generater(nn.Module):
    def __init__(self, opts):
        super(Generater, self).__init__()
        
        self.generate_linear = nn.Linear(
            2 * opts['dim_rnn'],
            opts['trg_voc_size']
        )

    '''
    def log_softmax(self, input_tensor):
        input_max, _ = torch.max(input_tensor, dim=2, keepdim=True)
        logsumexp_term = torch.log(
            1e-9 + torch.sum(
                torch.exp(
                    input_tensor - input_max,
                    ),
                dim=2,
                keepdim=True
                )
            ) + input_max
        return input_tensor - logsumexp_term
    '''
    def forward(self, input_tensor):
        assert len(input_tensor.size()) == 2
        return nn.functional.log_softmax(
            self.generate_linear(
                input_tensor
            )
        )
    



class Decoder(nn.Module):
    def __init__(self, opts):
        super(Decoder, self).__init__()
        
        self.embedding = Embedding(
            opts['trg_voc_size'],
            opts['dim_emb'],
            opts['dropout']
        )

        self.attn_layer = GlobalAttention(opts)
        
        self.dim_rnn = opts['dim_rnn']

        self.init_hidden = Variable(
            torch.zeros(
                1,
                2 * self.dim_rnn
            ),
            requires_grad=False
        )
        '''
        
        self.init_cell = Variable(
            torch.zeros(
                1,
                2 * self.dim_rnn
            ),
            requires_grad=False
        )

        self.linear_in = nn.Parameter(
            torch.FloatTensor(
                2 * self.dim_rnn, 
                2 * self.dim_rnn
            )
        )

        self.linear_out = nn.Parameter(
            torch.FloatTensor(
                2 * self.dim_rnn, 
                2 * self.dim_rnn
            )
        )
        '''


        self.lstm_cell = nn.LSTMCell(
            input_size=2 * opts['dim_rnn'] + opts['dim_emb'],
            hidden_size= 2 * opts['dim_rnn']
        )

        self.linear_in = nn.Linear(
            2 * self.dim_rnn, 
            2 * self.dim_rnn, 
            bias=False
        )

        self.generator = Generater(opts)

        self.dropout = nn.Dropout(p=opts['dropout']) 
        
    def forward(self, seq_context, src_mask, seq_trg, final_states_encoder):
        max_len_trg = seq_trg.size(0)
        batch_size = seq_trg.size(1)

        seq_trg_emb = self.embedding(seq_trg)
        seq_context_after_liner = self.linear_in(seq_context)
        
        '''
        prev_h = self.init_hidden.expand(batch_size, 2 * self.dim_rnn)
        prev_c = self.init_cell.expand(batch_size, 2 * self.dim_rnn)
        '''

        prev_h , prev_c = final_states_encoder

        prev_h = torch.cat(
            [
                prev_h[0:prev_h.size(0):2], 
                prev_h[1:prev_h.size(0):2]
            ], 
            dim=2
        )[0]

        prev_c = torch.cat(
            [
                prev_c[0:prev_c.size(0):2], 
                prev_c[1:prev_c.size(0):2]
            ], 
            dim=2
        )[0]

        h_list = []
        log_prob_list = []
        #print(prev_h.size())
        for i in range(1, max_len_trg):
            atten = self.attn_layer(
                seq_context,
                seq_context_after_liner, 
                src_mask,
                prev_h
            )
            #print(atten.size())    
            lstm_input = torch.cat(
                [
                    seq_trg_emb[i - 1],
                    atten
                ],
                dim=1
            )
            #print(lstm_input.size())
            prev_h = self.dropout(prev_h)
            prev_c = self.dropout(prev_c)
            #print(prev_c.size())
            #print(prev_h.size())
            #print(lstm_input.size())
            prev_h, prev_c = self.lstm_cell(lstm_input, (prev_h, prev_c))
            
            #h_list.append(prev_h.unsqueeze(0))
            log_prob_list.append(self.generator(prev_h).unsqueeze(0))
        
        seq_trg_log_prob = torch.cat(log_prob_list, dim=0)
        #seq_hidden_trg = torch.cat(h_list, dim=0)
        
        #seq_trg_log_prob = self.generator(seq_hidden_trg)

        return seq_trg_log_prob
    

class NMT(nn.Module):
    def __init__(self, opts):
        super(NMT, self).__init__()
        self.encoder = Encoder(opts)
        self.decoder = Decoder(opts)
    
    def forward(self,         
        src_batch,
        trg_batch,
        src_mask,
        trg_mask):
        seq_context, final_states = self.encoder(src_batch, src_mask)
        seq_trg_log_prob = self.decoder(seq_context, src_mask, trg_batch, final_states)

        return seq_trg_log_prob
    
    def load_param(self, path):
        with open(path, 'rb') as f:
            params = torch.load(f)
        for key, value in params.iteritems():
            print(key, value.size())
            
        # encoder param
        self.encoder.embedding.look_up.weight.data = params['encoder.embeddings.emb_luts.0.weight']

        self.encoder.bi_lstm.weight_hh_l0.data = params['encoder.rnn.weight_hh_l0']
        self.encoder.bi_lstm.bias_hh_l0.data = params['encoder.rnn.bias_hh_l0']        
        self.encoder.bi_lstm.weight_ih_l0.data = params['encoder.rnn.weight_ih_l0']
        self.encoder.bi_lstm.bias_ih_l0.data = params['encoder.rnn.bias_ih_l0'] 

        self.encoder.bi_lstm.weight_hh_l0_reverse.data = params['encoder.rnn.weight_hh_l0_reverse']
        self.encoder.bi_lstm.bias_hh_l0_reverse.data = params['encoder.rnn.bias_hh_l0_reverse']        
        self.encoder.bi_lstm.weight_ih_l0_reverse.data = params['encoder.rnn.weight_ih_l0_reverse']
        self.encoder.bi_lstm.bias_ih_l0_reverse.data = params['encoder.rnn.bias_ih_l0_reverse'] 
        
        # decoder param
        self.decoder.embedding.look_up.weight.data = params['decoder.embeddings.emb_luts.0.weight']

        self.decoder.lstm_cell.weight_hh.data = params['decoder.rnn.layers.0.weight_hh']
        self.decoder.lstm_cell.bias_hh.data = params['decoder.rnn.layers.0.bias_hh']
        self.decoder.lstm_cell.weight_ih.data = params['decoder.rnn.layers.0.weight_ih']
        self.decoder.lstm_cell.bias_ih.data = params['decoder.rnn.layers.0.bias_ih']

        self.decoder.linear_in.weight.data = params['decoder.attn.linear_in.weight'].transpose(0, 1)
        self.decoder.attn_layer.linear_out.weight.data = params['decoder.attn.linear_out.weight']

        self.decoder.generator.generate_linear.weight.data = params['0.weight']
        self.decoder.generator.generate_linear.bias.data = params['0.bias']
        
        
        
    
