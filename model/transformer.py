from model.encoder import *
from model.transformer_modules import *
from torch import nn

class Transformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_seq_length, dropout):
        super(Transformer, self).__init__()
        
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx = 0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder = Encoder(max_seq_length,dim_feedforward=dim_feedforward, d_model = d_model,n_layers=num_layers,heads = num_heads)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward = dim_feedforward,  dropout=0.1, batch_first=True),
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):

        tgt_embedded = self.decoder_embedding(tgt).transpose(0,1)
        
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt).transpose(0,1)).transpose(0,1))

        enc_output = self.encoder(src)

        dec_output = self.decoder(tgt_embedded,enc_output)

        return self.fc(dec_output)
    


if __name__ == "__main__":
    transformer = Transformer(5000,512,8,6,512*4,100,0.1)

    input_img = torch.randn(3,3,256,256)

    target = torch.randint(5000, (3,100))

    print(transformer(input_img,target).shape)