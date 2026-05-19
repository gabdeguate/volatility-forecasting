import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size= hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first= True)
        self.output = nn.Linear(in_features=hidden_size, out_features = 1)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # return shape: (batch_size, 1)
        
        out, _ =  self.lstm(x)
        
        pred = self.output(out[:,-1,:])
        
        return pred
    
        
        