import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import random
import time

NUM_EPOCHS =11

# PREPROCESSING 

# Poisson Normalization
class PoissonNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B,1,H,W)
        mu = x.mean(dim=(2,3), keepdim=True)
        return (x - mu) / torch.sqrt(mu + self.eps)

# Background Filtering
class HighPass(nn.Module):
    def __init__(self, kernel_size=21, sigma=7.0):
        super().__init__()
        self.padding = kernel_size // 2

        # fixed Gaussian kernel
        coords = torch.arange(kernel_size) - kernel_size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None,None])

    def forward(self, x):
        blur = F.conv2d(x, self.kernel, padding=self.padding)
        return x - blur


def highpass(x):
    return x - F.avg_pool2d(x, 5, stride=1, padding=2)

class FFTFeatures(nn.Module):
    def forward(self, x):
        # x: (B,1,H,W)
        x = x.squeeze(1)
        # CHANGE: If this gets changed the also the downscaling in REALSPACE ENCODER
        x = F.avg_pool2d(x, 2) 
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

        mag = torch.abs(fft)
        mag = torch.log(mag + 1e-6)
        phase = torch.angle(fft)

        # Bragg-peak enhanced map
        thresh = torch.quantile(mag.flatten(1), 0.99, dim=1).view(-1,1,1)
        peaks = F.relu(mag - thresh)

        # stack channels
        return torch.stack([mag, phase, peaks], dim=1)

# CNN Building Block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.net(x)




# ENCODERS

# Real Space Encodder
class RealEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(1, 32)
        self.c2 = ConvBlock(32, 64)

    def forward(self, x):
        x = self.c1(x)
        x = F.max_pool2d(x, 2)
        x = self.c2(x)
        x = F.max_pool2d(x, 2)
        # CHANGE: Downscaling the image also change in FFT FEATURES
        x = F.max_pool2d(x, 2)
        return x

# FFT Encdoer 
class FFTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 64)

    def forward(self, x):
        x = self.c1(x)
        x = F.max_pool2d(x, 2)
        x = self.c2(x)
        x = F.max_pool2d(x, 2)
        return x

# DECODER

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(128, 64)
        self.c2 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.c1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.c2(x)
        return self.out(x)
    
# MAIN MODEL
class TEMLatticeSeparator(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = PoissonNorm()
        self.hp = HighPass()

        self.fft = FFTFeatures()

        self.real_enc = RealEncoder()
        self.fft_enc = FFTEncoder()

        self.fuse = ConvBlock(128, 128)

        self.dec1 = Decoder()
        self.dec2 = Decoder()

    def forward(self, x):
        H, W = x.shape[-2:]
        # preprocessing
        x = self.norm(x)
        x = self.hp(x)

        # encoders
        r = self.real_enc(x)
        f = self.fft_enc(self.fft(x))

        # fuse
        z = torch.cat([r, f], dim=1)
        z = self.fuse(z)

        # decoders
        y1 = self.dec1(z)
        y2 = self.dec2(z)

        # Upscaling 
        y1 = F.interpolate(y1, size=(H, W), mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, size=(H, W), mode="bilinear", align_corners=False)

        # CHANGE: Clamping
        y1 = torch.clamp(y1, -5, 5)
        y2 = torch.clamp(y2, -5, 5)

        return y1, y2

# Loss Function
def reconstruction_loss(x, y1, y2):
    return F.mse_loss(y1 + y2, x)

# FFT Decorrelation 
def fft_decorrelation(y1, y2):
    with torch.no_grad():
        f1 = torch.abs(torch.fft.fft2(y1.squeeze(1)))
        f2 = torch.abs(torch.fft.fft2(y2.squeeze(1)))
        return torch.mean(f1 * f2)

# Total loss
def total_loss(x, y1p, y2p, y1, y2, w_rec, w_fft=0.1, w_decorr=0.1):

    L_real = F.mse_loss(y1p, y1) + F.mse_loss(y2p, y2)
    L_rec  = reconstruction_loss(x, y1p, y2p)

    loss = L_real + w_rec * L_rec
    # loss = L_real + w_rec * L_rec + 0.3 * L_y2_hp
    # ADD L_dec in 
    del L_real, L_rec
    return loss


# Image Loader
def load_png_tensor(path):
    """
    Loads a PNG as a (1, H, W) float32 tensor suitable for TEM data.
    """
    img = Image.open(path).convert("F")  # force float32 grayscale
    arr = np.array(img, dtype=np.float32)

    # Remove arbitrary PNG scaling / offset
    arr -= arr.min()
    arr /= (arr.mean() + 1e-6)

    # (1, H, W)
    return torch.from_numpy(arr).unsqueeze(0)

# Dataset Creator
class TEMLatticeDataset(Dataset):
    def __init__(self, root_dir):
        #root_dir = Path(os.getcwd())
        folder_path = Path(os.getcwd())
        root_dir = Path(root_dir)
        root_dir = folder_path / root_dir

        layer1_dir = root_dir / "layer1"
        layer2_dir = root_dir / "layer2"
        moire_dir  = root_dir / "moire"
        print(layer1_dir)
        # --- Load untilted reference (must be exactly one image) ---
       
        self.layer1_files = sorted(layer1_dir.glob("sample_*.png"))
        # --- Load paired datasets ---
        # CHANGE: CAN BE SUBSCRIPTED TO SHORTEN RUNTIME
        self.layer2_files = sorted(layer2_dir.glob("sample_*.png"))
        self.moire_files  = sorted(moire_dir.glob("sample_*.png"))

        # Optional: enforce identical filenames
        for f2, fm in zip(self.layer2_files, self.moire_files):
            assert f2.name == fm.name, \
                f"Filename mismatch: {f2.name} vs {fm.name}"

    def __len__(self):
        return len(self.moire_files)

    def __getitem__(self, idx):
        X  = load_png_tensor(self.moire_files[idx])   # combined image
        Y1=load_png_tensor(self.layer1_files[idx])                  
        Y2 = load_png_tensor(self.layer2_files[idx])  # tilted lattice

        return X, Y1, Y2




@torch.no_grad()
def visualize_tilted_vs_prediction_grid(model, dataset, device, n, runname, epoch):
    """
    Displays a 2xN grid:
      Top row    : ground-truth tilted lattice (layer2)
      Bottom row : predicted layer2 from model
    """
    model.eval()

    # Randomly choose indices
    indices = random.sample(range(len(dataset)), n)

    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))

    for col, idx in enumerate(indices):
        # Load data
        X, _, Y2 = dataset[idx]

        # Model prediction
        Xb = X.unsqueeze(0).to(device)
        _, y2p = model(Xb)

        # Move to CPU for plotting
        gt = Y2.squeeze().cpu()
        pred = y2p.squeeze().cpu()

        # --- Top row: ground truth tilted lattice ---
        axes[0, col].imshow(gt, cmap="gray")
        axes[0, col].set_title(f"GT layer2\nidx={idx}")
        axes[0, col].axis("off")

        # --- Bottom row: predicted layer2 ---
        axes[1, col].imshow(pred, cmap="gray")
        axes[1, col].set_title("Predicted layer2")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[1, 0].set_ylabel("Prediction", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"run_{runname}_results/outputs/output_sample_at_epoch_{epoch}.png")






# TRAINING LOOP


# DEVICE CHECK
if __name__ == "__main__":
    runname = input("Enter Run Name: ")


    base_dir = f"run_{runname}_results"

    os.makedirs(f"{base_dir}/weights", exist_ok=True)
    os.makedirs(f"{base_dir}/outputs", exist_ok=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)
    model = TEMLatticeSeparator().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)


    dataset = TEMLatticeDataset("simple_dataset")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,   
        pin_memory=False
    )

    
    for epoch in range(NUM_EPOCHS):
        if epoch == 5:
            for g in opt.param_groups:
                g["lr"] *= 0.5
        count = 0
        for X, Y1, Y2 in dataloader:
            X  = X.to(device)
            Y1 = Y1.to(device)
            Y2 = Y2.to(device)

            y1p, y2p = model(X)

            # CHANGE: w_rec can be increased
            loss = total_loss(X, y1p, y2p, Y1, Y2, w_rec = min(1.0, epoch / 3))
            y2loss = F.mse_loss(y2p, Y2).item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if count%10==0:
                print(count)
            count+=1

        save_path = f"run_{runname}_results/weights/tem_lattice_separator_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
        visualize_tilted_vs_prediction_grid(model, dataset, device, 5, runname, epoch)
        print(f"Epoch {epoch}: total loss={loss.item():.4f}")
        print(f"Epoch {epoch}: y2 loss={y2loss:.4f}")


        if device.type == "mps":
                torch.mps.empty_cache()




