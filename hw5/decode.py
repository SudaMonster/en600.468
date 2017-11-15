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
parser.add_argument("--model", required=True,
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

    nmt = torch.load(options.model)

    if use_cuda > 0:
        nmt.cuda()
    else:
        nmt.cpu()
    
    nmt.eval()
  
    for sentece in src_test:
        
        seq_context, final_states = nmt.encoder(sentece)
        
        seq_context_after_liner = nmt.decoder.linear_in(seq_context)
        
        prev_h , prev_c = final_states

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


        prev_word = '<s>'

        word_list = []
        
        while i < max_len:
            pre_emb = nmt.decoder.embedding(pre_v)
            atten = nmt.decoder.attn_layer(
                seq_context,
                seq_context_after_liner, 
                ones,
                prev_h
            )
            
            lstm_input = torch.cat(
                [
                    prev_emb,
                    atten
                ],
                dim=1
            )
            
            prev_h, prev_c = nmt.decoder.lstm_cell(lstm_input, (prev_h, prev_c))
            
            log_prob = nmt.decoder.generator(prev_h)

            _, prev_word = torch.max(log_prob)
        
            if prev_word == '<\s>':
                break
            else:
                word_list.append(trg_vocab.itos(prev_word))
            
            i+=1

        with open(options.output, 'a+') as f:
            f.write(' '.join(word_list) + '\n')

        sys.stderr.write(' '.join(word_list) + '\n')
        


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
