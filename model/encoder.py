from torch import nn
import torch
import torch.nn.functional as F
import torchvision
from model.transformer_modules import EncoderLayer

class ImageEncoder(nn.Module):
    def __init__(self, sequence_len, encode_size=14, embed_dim=512):
        """
        param:
        encode_size:    encoded image size.
                        int

        embed_dim:      encoded images features dimension
                        int
        """
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        # pretrained ImageNet ResNet-101
        # Remove last linear and pool layers
        resnet = torchvision.models.resnet101()
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # disable fine-tuning for pretrained resnet
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=embed_dim,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(encode_size * encode_size, sequence_len)

        # Resize images, use 2D adaptive max pooling
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)

    def forward(self, images: torch.Tensor):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # image_size = [B, 3, h, w]
        B = images.size()[0]

        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        out = self.resnet(images)

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))

        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->
        #       [B, embed_size=512, encode_size=14, encode_size=14] ->
        #           [B, 512, 196] -> [B, 196, 512]
        out = self.adaptive_resize(out)
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)

        out = out.transpose(1,2)

        out = self.relu2(out)

        out = self.fc2(out)

        out = out.transpose(1,2)

        # [B, 196, 512] - [batch, sequence_length, embedding_dim] 
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the tuning for blocks 2 through 4.
        """

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Encoder(nn.Module):
    """
    Encoder which consists from two main blocks: 
    Image Encoder and classic notion of Encoder defined in Transformer's article "Attention is all you need 2017"
    """
    def __init__(self, seq_len, d_model=512, dim_feedforward = 2048, n_layers=6, heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        assert d_model % heads == 0

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        self.image_encoder = ImageEncoder(seq_len, embed_dim=d_model)

        # multi-layers transformer blocks, deep network
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first = True),
            num_layers=n_layers
        )

    def forward(self, batch_images):
        # batch x 3 x height x width
        out = self.image_encoder(batch_images)

        # running over multiple transformer blocks
        # batch x seq_len x d_model
        return self.encoder(out)


if __name__ == "__main__":

    encoder = Encoder(100,512)

    test_image = torch.randn(1,3,256,256)

    print(test_image.shape)


    

    
