import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
import math
from model import NMT
import sys
import os
import heapq
#from example_module import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Decoding")
parser.add_argument("--data_file", required=True,
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = de)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = en)")
parser.add_argument("--models", required=True, nargs='+',
                    help="Location to dump the models.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--width", default=12, type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


def next_log_prob(nmts, prev_w, prev_h_, prev_c_, output_, seq_context):
    
    num_model = len(nmts)

    prob = 0
    
    output = list(output_)
    prev_h = list(prev_h_)
    prev_c = list(prev_c_)


    for k, nmt in enumerate(nmts):
                
        emb_w = nmt.decoder.embedding(prev_w)
        #print emb_w, output
        lstm_input = torch.cat(
            [
                emb_w,
                output[k]
            ],
            dim=1
        )
                #print type(prev_h), len(prev_h), k
        prev_h[k], prev_c[k] = nmt.decoder.lstm_cell(
            lstm_input, 
            (prev_h[k], prev_c[k])
        )
        
        output[k] = nmt.decoder.attn_layer(
            prev_h[k],
            seq_context[k],
            None
        )

        output[k] = nmt.decoder.dropout(output[k])
            
        prob += torch.exp(nmt.generator(output[k]))
        
    return torch.log(prob / num_model), prev_h, prev_c, output



def main(options):
    use_cuda = (len(options.gpuid) >= 1)

    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
    trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

    trg_vocab_size = len(trg_vocab)
    src_vocab_size = len(src_vocab)
    
    nmts = []
    for model in options.models:
        nmt = torch.load(model)
        if use_cuda > 0:
            nmt.cuda()
        else:
            nmt.cpu()
        nmt.eval()
        nmts.append(nmt)
  
    os.system("rm -f {}".format(options.models[0] + '.ensemble.beam.out'))
    for src_sent_ in src_test:
        
        src_sent = Variable(src_sent_[:, None]) 
        
        if use_cuda > 0:
            src_sent = src_sent.cuda()
        
        seq_context = []
        #final_states = []
        prev_h = []
        prev_c = []
        output = []
        
        
        for nmt in nmts:
            _seq_context, _final_states = nmt.encoder(src_sent, None)
            seq_context.append(_seq_context)
            #final_states.append(final_states_)

            _prev_h , _prev_c = _final_states
        
            # initial states
            _prev_h = torch.cat(
                [
                    _prev_h[0:_prev_h.size(0):2], 
                    _prev_h[1:_prev_h.size(0):2]
                ], 
                dim=2
            )[0]

            _prev_c = torch.cat(
                [
                    _prev_c[0:_prev_c.size(0):2], 
                    _prev_c[1:_prev_c.size(0):2]
                ], 
                dim=2
            )[0]

            _output = Variable(_seq_context.data.new(1, nmt.dim_rnn * 2))

            prev_h.append(_prev_h)
            prev_c.append(_prev_c)
            output.append(_output)


        w = Variable(
            torch.LongTensor(1)
        )
        
        if use_cuda > 0:
            w = w.cuda()
        

        w[0] = trg_vocab.stoi['<s>']
        w_list = []

        i = 0
        decoder_output_list = []
        
        hyp_words = [[w]  for _ in range( options.width)]
        hyp_hiddens = [(prev_h, prev_c, output) for _ in range(options.width)]
        hyp_scores = [0 for _ in range( options.width)]
        
        all_done = 1
        while i < 80 :
            #print hyp_words[0]
            #raw_input()
            i += 1
            #print i
            #for l in range(options.width):
            #    best_words = [trg_vocab.itos[w.data[0]] for w in hyp_words[l]]
            #    trans_sent =  u' '.join(best_words).encode('utf-8')
            #    print trans_sent    
            #raw_input()

            new_hyps = []
       
            all_done = 1
            #import pdb; pdb.set_trace()
            for hyp_idx in range(options.width):
                _prev_w = hyp_words[hyp_idx][-1]
                _prev_h, _prev_c, _output = hyp_hiddens[hyp_idx]
                _prev_score = hyp_scores[hyp_idx]

                if _prev_w.data[0] == trg_vocab.stoi['</s>']:
                    heapq.heappush(
                        new_hyps,
                        (
                            _prev_score / len(hyp_words[hyp_idx]), 
                            (
                                hyp_words[hyp_idx], 
                                _prev_h,
                                _prev_c,
                                _output,
                                _prev_score
                            )
                        )
                    )
                    
                #print type(hyp_hiddens), len(hyp_hiddens)
                else:
                    all_done = 0
                    
                    _log_prob, _h, _c, _o = next_log_prob(
                        nmts,
                        _prev_w,
                        _prev_h,
                        _prev_c,
                        _output,
                        seq_context
                    )

                    log_prob_topk, w_topk = torch.topk(_log_prob, options.width)
                    #print log_prob_topk, w_topk
                    #print w_topk
                    #raw_input()

                    for j in range(options.width):
                        new_score = -log_prob_topk[0][j].data[0] + _prev_score
                        if new_score / (1 + len(hyp_words[hyp_idx])) in [x[0] for x in new_hyps]:
                            continue
                        heapq.heappush(
                            new_hyps,
                            (
                                new_score / (1 + len(hyp_words[hyp_idx])),
                                (
                                    hyp_words[hyp_idx] + [w_topk[0][j]], 
                                    _h,
                                    _c,
                                    _o,
                                    new_score
                                )
                            )
                        )
            
            #for hyp_ in new_hyps:
            #    s, (words,_,_,_,_) = hyp_
            #    print s, u' '.join([trg_vocab.itos[w.data[0]] for w in words]).encode('utf-8')
            #print all_done
            #raw_input()


            if all_done == 1:
                break
            ''' 
            if i == 1:
                for k in range(options.width):
                    s, (words, prev_h, prev_c, output_) = new_hyps[k * options.width]
                    hyp_words[k] = words
                    hyp_hiddens[k] = (prev_h, prev_c, output_)
                    hyp_scores[k] = s

                continue
            '''
            for k in range(options.width):
                s, (words, prev_h, prev_c, output_, score) = heapq.heappop(new_hyps)
                hyp_words[k] = words
                hyp_hiddens[k] = (prev_h, prev_c, output_)
                hyp_scores[k] = score
        
        best_words = [trg_vocab.itos[w.data[0]] for w in hyp_words[0]]
        trans_sent =  u' '.join(best_words[1:-1]).encode('utf-8')
    
        with open(options.models[0] + '.ensemble.beam.out', 'a+') as f:
            f.write(trans_sent + '\n')

        sys.stderr.write(trans_sent + '\n')
        


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
