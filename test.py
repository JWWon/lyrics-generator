import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import os
import argparse

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden state')
    parser.add_argument('--num_layers', type=int, default=3, help='Num of layers in RNN layer stack')
    parser.add_argument('--output_seq_len', type=int, default=500, help='Total num of characters in output test sequence')
    parser.add_argument('--start_text', type=str, default=None, help='Text which starts the generator')
    parser.add_argument('--data_path', type=str, default='./data/ballad/')
    args = parser.parse_args()
    test(args)

def test(args):
    # Load text file
    data = open(args.data_path + "input.txt", 'r').read()
    vocab = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(vocab)
    print("################################################################################")
    print(f"## Data {args.data_path} has {data_size} characters, and {vocab_size} unique vocabularies ##")
    print("################################################################################")
    print("")

    # Map vocab as dict of character and index
    char_to_idx = { char: idx for idx, char in enumerate(vocab) }
    idx_to_char = { idx: char for idx, char in enumerate(vocab) }

    # Convert data from to indices
    data = list(data)
    for idx, char in enumerate(data):
        data[idx] = char_to_idx[char]

    # Data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)

    # Create model instance and load pretrained weight
    model = Model(input_size=vocab_size, output_size=vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    print(f"Loading pretrained weight from {args.data_path}...")
    try:
        model.load_state_dict(torch.load(args.data_path + "last.pth"), strict=False)
    except:
        sys.exit("Model should be trained before testing")
    print("Successfully load weight from file")
    print("")

    # initialize variables
    data_ptr = 0
    hidden_state = None

    if args.start_text != None:
        raw_texts = args.start_text.split(" ")
        tensor_items = [data[char_to_idx[text]] for text in enumerate(raw_texts)]   # TODO: Handle exceptions
        input_seq = torch.cat(tensor_items)
    else:
        rand_idx = np.random.randint(data_size - 11)
        input_seq = data[rand_idx:rand_idx+9] # 10 characters

        # Compute last hidden state of the sequence
        _, hidden_state = model(input_seq, hidden_state)

        # Next element is input
        input_seq = data[rand_idx+9:rand_idx+10]

    print(f"Start testing on {device}")
    print("")

    # Print input sequence
    seq_item = input_seq.item()
    print("'", end='')
    if type(seq_item) is str:
        print(idx_to_char[seq_item], end='')
    elif type(seq_item) is list:
        for seq_idx in enumerate(seq_item):
            print(idx_to_char[seq_idx], end='')
    print("' -> ", end='')

    while True:
        # Forward pass
        output, hidden_state = model(input_seq, hidden_state)

        # Construct categorical distribution and sample character
        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample().item()

        # Pring sample
        print(idx_to_char[index], end='')

        # Next input is current output
        input_seq[0][0] = index
        data_ptr += 1

        if data_ptr > args.output_seq_len:
            break

    print("\n----------------------------------")



if __name__ == '__main__':
    main()
