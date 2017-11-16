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

        self.linear_in = nn.Linear(2 * self.dim_rnn, 2 * self.dim_rnn, bias=False)
        self.linear_out = nn.Linear(4 * self.dim_rnn, 2 * self.dim_rnn, bias=False)
        self.softmax= nn.Softmax()
        self.tanh = nn.Tanh()
    
    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim
        h_s (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x tgt_len x src_len:
            raw attention scores for each src index
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)

        return torch.bmm(h_t, h_s_)

    def forward(
        self,
        trg_hidden,
        context,
        src_mask
    ):
        context = context.transpose(0, 1)
        trg_hidden = trg_hidden.unsqueeze(0).transpose(0, 1)
        
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = trg_hidden.size()

        # align shape (batch, t_len, s_len)
        align = self.score(trg_hidden, context)
        
        if src_mask is not None:
            src_mask = src_mask.transpose(0, 1)
            mask_ = src_mask.contiguous().view(batch, 1, sourceL)  # make it broardcastable
            align.data.masked_fill_(1 - mask_.data, -float('inf'))
            

        align_vectors = self.softmax(
            align.view(
                batch * targetL, 
                sourceL
            )
        )

        #print align_vectors
        #raw_input()

        align_vectors = align_vectors.view(batch, targetL, sourceL)

        c = torch.bmm(align_vectors, context)

        concat_c = torch.cat(
            [
                c, 
                trg_hidden
            ], 
            dim=2
        ).view(batch * targetL, dim*2)
        
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)

        attn_h = self.tanh(attn_h)

        return attn_h.squeeze(1)
        
        



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

        self.lstm_cell = nn.LSTMCell(
            input_size=2 * opts['dim_rnn'] + opts['dim_emb'],
            hidden_size= 2 * opts['dim_rnn']
        )

        self.linear_in = nn.Linear(
            2 * self.dim_rnn, 
            2 * self.dim_rnn, 
            bias=False
        )

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
        # initial states
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

        output = Variable(seq_context.data.new(batch_size, self.dim_rnn * 2))

        decoder_output_list = []
        log_prob_list = []

        for i in range(max_len_trg - 1):
            lstm_input = torch.cat(
                [
                    seq_trg_emb[i],
                    output
                ],
                dim=1
            )

            prev_h, prev_c = self.lstm_cell(
                lstm_input, 
                (prev_h, prev_c)
            )
            
            output = self.attn_layer(
                prev_h,
                seq_context,
                src_mask
            )

            output = self.dropout(output)
            
            decoder_output_list.append(output.unsqueeze(0))
        
        # decoder_output seq_len * batch * rnn_dim
        decoder_output = torch.cat(decoder_output_list, dim=0)

        return decoder_output
    

class NMT(nn.Module):
    def __init__(self, opts):
        super(NMT, self).__init__()
        self.dim_rnn = opts['dim_rnn']
        self.encoder = Encoder(opts)
        self.decoder = Decoder(opts)
        self.generator = Generater(opts)
    
    def forward(self,         
        src_batch,
        trg_batch,
        src_mask,
        trg_mask):
        seq_context, final_states = self.encoder(src_batch, src_mask)
        decoder_output = self.decoder(seq_context, src_mask, trg_batch, final_states)
        
        trg_len, batch_size, decoder_dim = decoder_output.size()
        
        seq_trg_log_prob = self.generator(
            decoder_output.view(trg_len * batch_size, -1)
        ).view(trg_len, batch_size, -1)

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

        #self.decoder.linear_in.weight.data = params['decoder.attn.linear_in.weight'].transpose(0, 1)
        self.decoder.attn_layer.linear_in.weight.data = params['decoder.attn.linear_in.weight']
        self.decoder.attn_layer.linear_out.weight.data = params['decoder.attn.linear_out.weight']

        self.generator.generate_linear.weight.data = params['0.weight']
        self.generator.generate_linear.bias.data = params['0.bias']
    
    def decode(self, src_sent, trg_vocab):
        #print src_sent
        #print type(src_sent)
        src_sent = Variable(src_sent[:, None])
        seq_context, final_states = self.encoder(src_sent, None)

        prev_h , prev_c = final_states
        
        # initial states
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

        output = Variable(seq_context.data.new(1, self.dim_rnn * 2))

        decoder_output_list = []
        log_prob_list = []

        w = Variable(
            torch.LongTensor(1)
        )
        w[0] = trg_vocab.stoi['<s>']

        w_list = []

        i = 0

        while i < 100 :
            emb_w = self.decoder.embedding(w)
            #print emb_w, output
            lstm_input = torch.cat(
                [
                    emb_w,
                    output
                ],
                dim=1
            )

            prev_h, prev_c = self.decoder.lstm_cell(
                lstm_input, 
                (prev_h, prev_c)
            )
            
            output = self.decoder.attn_layer(
                prev_h,
                seq_context,
                None
            )

            output = self.decoder.dropout(output)
            
            log_prob = self.generator(output)

            _, w = torch.max(log_prob, dim=1)

            if w.data[0] == trg_vocab.stoi['</s>']:
                break
            
            w_list.append(trg_vocab.itos[w.data[0]])

            
        
        return ' '.join(w_list)
        

        
        
    
