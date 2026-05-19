import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model, seq_len, nhead, num_layers):
        super().__init__() 
        
        self.input = nn.Linear(2, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.pos_encoding = nn.Embedding(seq_len + 1, d_model)
        self.softplus = nn.Softplus()
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input(x)
        #Expand cls token 
        cls_token = self.cls.expand(batch_size, -1, -1)
        
        x = torch.cat((cls_token,x), dim = 1)
        
        positions = torch.arange(0, x.size(1), device= x.device).unsqueeze(0).expand(batch_size,-1)
        x = x + self.pos_encoding(positions)
        
        x = self.encoder(x)
        
        cls_output = x[:, 0, :] # CLS token contains summary of whole sequence
        
        out = self.output(cls_output)
        out = self.softplus(out) 
        
        return out
        