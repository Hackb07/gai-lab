import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=100, w_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim)
        )

    def forward(self, z):
        return self.net(z)

class StyledGenerator(nn.Module):
    def __init__(self, w_dim=100):
        super().__init__()
        self.fc = nn.Linear(w_dim, 256)
        self.conv = nn.ConvTranspose2d(256, 1, 4, 2, 1)

    def forward(self, w):
        x = self.fc(w).view(-1, 256, 1, 1)
        return torch.tanh(self.conv(x))

# Example
z = torch.randn(4, 100)
mapping = MappingNetwork()
generator = StyledGenerator()

w = mapping(z)
fake_images = generator(w)
print(fake_images.shape)


print("--------------------New images ----------------")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# --------------------------------------------------
# Mapping Network
# --------------------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=100, w_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim)
        )

    def forward(self, z):
        return self.net(z)

# --------------------------------------------------
# Styled Generator
# --------------------------------------------------
class StyledGenerator(nn.Module):
    def __init__(self, w_dim=100):
        super().__init__()
        self.fc = nn.Linear(w_dim, 256)
        self.conv = nn.ConvTranspose2d(256, 1, 4, 2, 1)

    def forward(self, w):
        x = self.fc(w).view(-1, 256, 1, 1)
        return torch.tanh(self.conv(x))

# --------------------------------------------------
# Generate Image
# --------------------------------------------------
z = torch.randn(4, 100)
mapping = MappingNetwork()
generator = StyledGenerator()

w = mapping(z)
fake_images = generator(w)

print("Generated image batch shape:", fake_images.shape)

# --------------------------------------------------
# Select ONE image
# --------------------------------------------------
# --------------------------------------------------
# Select ONE image
# --------------------------------------------------
image = fake_images[0]   # shape: [1, H, W]

# Detach from graph
image = image.detach()

# Convert from [-1,1] → [0,1]
image = (image + 1) / 2

# --------------------------------------------------
# Save Image
# --------------------------------------------------
from torchvision.utils import save_image
save_image(image, "stylegan_generated_image.png")

print("✅ Image saved as stylegan_generated_image.png")

# --------------------------------------------------
# Display Image (FIXED)
# --------------------------------------------------
import matplotlib.pyplot as plt

plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
plt.title("StyleGAN Generated Image")
plt.axis("off")
plt.show()
