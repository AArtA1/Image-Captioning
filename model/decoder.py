from transformer_modules import DecoderLayer
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, feedforward_dim, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, feedforward_dim, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, self_mask=self_mask, encoder_mask=encoder_mask)
        return x


if __name__ == "__main__":
    num_layers = 6
    d_model = 512
    num_heads = 8
    feedforward_dim = 2048
    seq_len = 20
    batch_size = 32

    decoder = Decoder(num_layers, d_model, num_heads, feedforward_dim)
    input_tensor = torch.rand(batch_size, seq_len, d_model)
    encoder_output = torch.rand(batch_size, seq_len, d_model)  # This is the output from the encoder
    output_tensor = decoder(input_tensor, encoder_output)
    
    assert output_tensor.size() == encoder_output.size() # Should be torch.Size([32, 20, 512])