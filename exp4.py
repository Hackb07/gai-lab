import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

class ProGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


z = torch.randn(8, 100, 1, 1)
G = ProGenerator()

fake_images = G(z)

print("Generated image shape:", fake_images.shape)


fake_images = (fake_images + 1) / 2


grid = make_grid(fake_images, nrow=4)

plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.title("Generated Images (ProGAN)")
plt.axis("off")
plt.show()


save_image(fake_images, "progan_generated_images.png", nrow=4)

print("✅ Images displayed and saved as progan_generated_images.png")
