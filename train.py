import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np

import json
import os
import argparse
import glob
import datetime

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
    parser.add_argument('--data_path', type=str, default='./data/ballad/', help='Directory which data and state dict exists')
    parser.add_argument('--save_name', type=str, default=None, help='Name of training result directory. Type of file should be {title}/last.pth')
    parser.add_argument('--dropout', type=float, default=0.3, help='Select dropout size')
    args = parser.parse_args()
    train(args)


def get_latest_state_dict(data_path: str):
    return glob.glob(data_path + "last*.pth")[-1]


def get_current_timestamp():
    return int(round(datetime.datetime.now().timestamp()))


def get_matrix_accuracy(truth, pred):
    truth_max = torch.argmax(truth, dim=0)
    pred_max = torch.argmax(pred, dim=0)

    match = torch.sum(truth_max == pred_max)

    return match.item() / truth_max.size(0)


def train(args):
        
    # Load text file
    data = open(args.data_path + "input.txt", 'r', encoding='utf-8').read() # 전체 데이터
    vocab = sorted(list(set(data)))         # 단어 리스트
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

    # Model instance
    model = Model(input_size=vocab_size, output_size=vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)

    # Load checkpoint if required
    if args.resume == True:
        print(f"Loading pretrained weight from {get_latest_state_dict(args.data_path)}...")
        model.load_state_dict(torch.load(get_latest_state_dict(args.data_path)))
        print("Successfully load weight from file")
        print("")

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()   # TODO: Find best loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # TODO: Find best optimizer function

    # Training loop
    save_name = args.save_name if args.save_name != None else f"{get_current_timestamp()}"
    save_path = args.data_path + save_name
    save_dict = save_path + '/last.pth'
    log_path = save_path + '/log'
    writer = SummaryWriter(log_path)
    writer.add_text('arguments', json.dumps(vars(args)))
    print(f"Start training on {device}, Save dict will be saved on {save_dict}")
    print(f"Arguments: {json.dumps(vars(args))}\n")
    for epoch in range(1, args.epochs + 1):
        data_ptr = np.random.randint(100)
        n = 0
        running_accuracy = 0
        running_loss = 0
        hidden_state = None

        while True:
            input_seq = data[data_ptr:data_ptr+args.seq_len]
            target_seq = data[data_ptr+1:data_ptr+args.seq_len+1]

            # forward
            output, hidden_state = model(input_seq, hidden_state)

            # accuracy
            running_accuracy += get_matrix_accuracy(torch.squeeze(output), torch.squeeze(target_seq))

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

        sum_accuracy = running_accuracy / n
        sum_loss = running_loss / n
        print(f"Epoch: {epoch} | Accuracy: {sum_accuracy:.6f} | Loss: {sum_loss:.8f}")
        
        writer.add_scalar("Accuracy/train", sum_accuracy, epoch)
        writer.add_scalar("Loss/train", sum_loss, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        
        torch.save(model.state_dict(), save_dict)

        # Sample / Generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = None

        # Random character from data to begin
        rand_index = np.random.randint(data_size - 1)
        input_seq = data[rand_index:rand_index+1]

        # Print sample text
        if epoch % 10 == 0:
            pred_text = ''
            while True:
                # forward
                output, hidden_state = model(input_seq, hidden_state)

                # Construct categorical distribution and sample a character
                output = F.softmax(torch.squeeze(output), dim=0)
                dist = Categorical(output)
                index = dist.sample().item()

                # Save sample
                pred_text += idx_to_char[index]

                # Next input is current output
                input_seq[0][0] = index
                data_ptr += 1

                if data_ptr > args.output_seq_len:
                    break
        
            print("\nPrediction\n")
            print(pred_text)

            writer.add_text("Pred", pred_text, epoch)
            writer.flush()

        print("-----------------------------------------")

    writer.close()
    os.system(f"tensorboard dev upload --logdir {log_path} --name 'train_{save_name}' --one_shot")

if __name__ == '__main__':
    main()
