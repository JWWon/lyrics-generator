import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=input_size)          # TODO: Find the best value of embedding_dim. Because it's a hyperparameter.
        self.net = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)   # TODO: We can switch 'LSTM' into better model
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.net(embedding, hidden_state)
        output = self.fc(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())
