import torch

import argparse

from utils import TextLoader

def main():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('--epoch_size', type=int, default=300, help='Size of epoch')
    parser.add_argument('--batch_size', type=int, default=60, help='Size of batch')
    parser.add_argument('--seq_length', type=int, default=100, help='Size of continuous sequences')
    parser.add_argument('--data_dir', type=str, default='./data/iu_lyrics', help='Path of training data')
    args = parser.parse_args()
    train(args)


def train(args):
    loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = loader.vocab_size


if __name__ == '__main__':
    main()
