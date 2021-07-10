import re

import torch
import torch.nn as nn
from torch import Tensor
from torchtext import vocab

from utils import get_text


def generate_translation_ext(src: Tensor, trg: Tensor,
                             model: nn.Module, 
                             SRC_vocab: vocab.Vocab, TRG_vocab: vocab.Vocab,
                             pattern=None):
    model.eval()

    with torch.no_grad():
        if model.forward.__code__.co_argcount == 4:
            output = model(src, trg, 0) #turn off teacher forcing
        else:
            output = model(src, trg)

        output = output.argmax(dim=-1).cpu().numpy()

        original = get_text(list(src[:, 0].cpu().numpy()), SRC_vocab)
        translation = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
        generated = get_text(list(output[1:, 0]), TRG_vocab)
        
        original_text = ' '.join(original)
        translation_text = ' '.join(translation)
        generated_text = ' '.join(generated)
        
        if pattern:
            original_text = re.sub(pattern, '', original_text)
            translation_text = re.sub(pattern, '', translation_text)
            generated_text = re.sub(pattern, '', generated_text)

        print('Original:\t{}'.format(original_text))
        print('Translation:\t{}'.format(translation_text))
        print('Generated:\t{}'.format(generated_text))
        
    print()


def generate_translation_emb(src: Tensor, trg: Tensor,
                             model: nn.Module, TRG_vocab: vocab.Vocab):
    model.eval()

    with torch.no_grad():
        if model.forward.__code__.co_argcount == 4:
            output = model(src, trg, 0) #turn off teacher forcing
        else:
            output = model(src, trg)

        output = output.argmax(dim=-1).cpu().numpy()

        translation = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
        generated = get_text(list(output[1:, 0]), TRG_vocab)

        print('Translation:\t{}'.format(' '.join(translation)))
        print('Generated:\t{}'.format(' '.join(generated)))

    print()
