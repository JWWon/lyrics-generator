import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import argparse

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden state')
    parser.add_argument('--seq_len', type=int, default=100, help='Length of RNN sequence')
    parser.add_argument('--num_layers', type=int, default=3, help='Num of layers in RNN layer stack')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum numbers of epochs')
    parser.add_argument('--output_seq_len', type=int, default=200, help='Total num of characters in output test sequence')
    parser.add_argument('--resume', type=bool, default=False, help='Load weights from save_file to resume training')
    parser.add_argument('--save_file', type=str, default='./pretrained/iu_lyrics.pth')
    parser.add_argument('--data_file', type=str, default='./data/iu_lyrics.txt')
    args = parser.parse_args()
    train(args)

def train(args):
    # Load text file
    data = open(args.data_file, 'r').read() # 전체 데이터
    vocab = sorted(list(set(data)))         # 단어 리스트
    data_size, vocab_size = len(data), len(vocab)
    print("################################################################################")
    print(f"## Data {args.data_file} has {data_size} characters, and {vocab_size} unique vocabularies ##")
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

    # Model instance
    model = Model(input_size=vocab_size, output_size=vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)

    # Load checkpoint if required
    if args.resume == True:
        print(f"Loading pretrained weight from {args.save_file}...")
        model.load_state_dict(torch.load(args.save_file))
        print("Successfully load weight from file")
        print("")

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()   # TODO: Find best loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # TODO: Find best optimizer function

    # Training loop
    print(f"Start training on {device}")
    print("")
    for epoch in range(1, args.epochs + 1):
        data_ptr = np.random.randint(100)
        n = 0
        running_loss = 0
        hidden_state = None

        while True:
            input_seq = data[data_ptr:data_ptr+args.seq_len]
            target_seq = data[data_ptr+1:data_ptr+args.seq_len+1]

            # forward
            output, hidden_state = model(input_seq, hidden_state)

            # loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update parameters
            data_ptr += args.seq_len
            n += 1

            if data_ptr + args.seq_len + 1 > data_size:
                break

        print("Epoch: {0} | Loss: {1:.8f}".format(epoch, running_loss / n))
        print("")
        torch.save(model.state_dict(), args.save_file)

        # Sample / Generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = None

        # Random character from data to begin
        rand_index = np.random.randint(data_size - 1)
        input_seq = data[rand_index:rand_index+1]

        # Print sample text
        while True:
            # forward
            output, hidden_state = model(input_seq, hidden_state)

            # Construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample().item()

            # Print sample
            print(idx_to_char[index], end='')

            # Next input is current output
            input_seq[0][0] = index
            data_ptr += 1

            if data_ptr > args.output_seq_len:
                break

        print("\n-----------------------------------------")


if __name__ == '__main__':
    main()
