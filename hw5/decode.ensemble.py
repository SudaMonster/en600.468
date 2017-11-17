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
  
    os.system("rm -f {}".format(options.models[0] + '.ensemble.out'))
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
        
        while i < 80 :
            prob = 0
            i += 1
            for k, nmt in enumerate(nmts):
                emb_w = nmt.decoder.embedding(w)
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

            _, w = torch.max(prob, dim=1)

            if w.data[0] == trg_vocab.stoi['</s>']:
                break
            
            w_list.append(trg_vocab.itos[w.data[0]])
            
        
        trans_sent =  u' '.join(w_list).encode('utf-8')
    
        with open(options.models[0] + '.ensemble.out', 'a+') as f:
            f.write(trans_sent + '\n')

        sys.stderr.write(trans_sent + '\n')
        


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
