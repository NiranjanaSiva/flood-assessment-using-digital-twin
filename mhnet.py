import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Layer Normalization for image feature maps
# ============================================================

class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        # x = [B, C, H, W]

        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        x = x.permute(0, 3, 1, 2)

        return x


# ============================================================
# MDTA
# Multi-Dconv Head Transposed Attention
#
# This is the attention component used by Restormer.
# ============================================================

class MDTA(nn.Module):

    def __init__(self, channels, heads=4):

        super().__init__()

        self.heads = heads

        self.temperature = nn.Parameter(
            torch.ones(heads, 1, 1)
        )

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=False
        )

        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False
        )

        self.project_out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        b, c, h, w = x.shape

        qkv = self.qkv(x)

        qkv = self.qkv_dw(qkv)

        q, k, v = qkv.chunk(3, dim=1)

        # Divide channels into attention heads

        q = q.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        k = k.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        v = v.reshape(
            b,
            self.heads,
            c // self.heads,
            h * w
        )

        # Normalize query and key

        q = F.normalize(q, dim=-1)

        k = F.normalize(k, dim=-1)

        # Attention

        attention = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention = attention * self.temperature

        attention = torch.softmax(
            attention,
            dim=-1
        )

        out = torch.matmul(
            attention,
            v
        )

        out = out.reshape(
            b,
            c,
            h,
            w
        )

        out = self.project_out(out)

        return out


# ============================================================
# GDFN
# Gated-Dconv Feed Forward Network
#
# Restormer component.
# ============================================================

class GDFN(nn.Module):

    def __init__(
        self,
        channels,
        expansion=2.66
    ):

        super().__init__()

        hidden = int(channels * expansion)

        self.project_in = nn.Conv2d(
            channels,
            hidden * 2,
            kernel_size=1
        )

        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2
        )

        self.project_out = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1
        )

    def forward(self, x):

        x = self.project_in(x)

        x = self.dwconv(x)

        x1, x2 = x.chunk(2, dim=1)

        x = F.gelu(x1) * x2

        x = self.project_out(x)

        return x


# ============================================================
# Restormer Block
#
# Paper Eq. (5):
#
# Fout = fGDFN(LN(fMDTA(LN(Fin))))
#
# Residual connections are used in the standard Restormer
# implementation.
# ============================================================

class RestormerBlock(nn.Module):

    def __init__(
        self,
        channels,
        heads=4,
        dropout=0.0
    ):

        super().__init__()

        self.norm1 = LayerNorm2d(channels)

        self.mdta = MDTA(
            channels,
            heads
        )

        self.norm2 = LayerNorm2d(channels)

        self.gdfn = GDFN(
            channels
        )

        self.dropout = nn.Dropout2d(
            dropout
        )

    def forward(self, x):

        # MDTA

        x = x + self.dropout(
            self.mdta(
                self.norm1(x)
            )
        )

        # GDFN

        x = x + self.dropout(
            self.gdfn(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# Convolutional Embedding
#
# Paper Eq. (1):
#
# Fembed = BN(LeakyReLU(Conv(I)))
# ============================================================

class ConvEmbedding(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=32
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.activation = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.activation(x)

        return x


# ============================================================
# Space-to-Depth
#
# Paper Eq. (2):
#
# Frs = space_to_depth(Fr)
# ============================================================

class SpaceToDepth(nn.Module):

    def __init__(self, block_size=2):

        super().__init__()

        self.block_size = block_size

    def forward(self, x):

        b, c, h, w = x.shape

        s = self.block_size

        # Make dimensions divisible by block size

        h_new = h // s * s
        w_new = w // s * s

        x = x[:, :, :h_new, :w_new]

        x = x.reshape(
            b,
            c,
            h_new // s,
            s,
            w_new // s,
            s
        )

        x = x.permute(
            0,
            1,
            3,
            5,
            2,
            4
        )

        x = x.reshape(
            b,
            c * s * s,
            h_new // s,
            w_new // s
        )

        return x


# ============================================================
# Resize Down
#
# Paper uses convolution/transposed-convolution based
# resizing modules rather than arbitrary interpolation.
# ============================================================

class ResizeDown(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Resize Up
#
# 2x upsampling using transposed convolution.
# ============================================================

class ResizeUp(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Feature Masking
#
# Paper Eq. (4):
#
# Fmask = Mask_to_token(Random_Mask(Fin,T))
#
# Masking is used ONLY during training.
#
# Selected paper setting:
# random noise
# mask ratio = 0.25
# same ratio at all 3 scales
# ============================================================

class FeatureMasking(nn.Module):

    def __init__(
        self,
        channels,
        mask_ratio=0.25,
        patch_size=4
    ):

        super().__init__()

        self.mask_ratio = mask_ratio

        self.patch_size = patch_size

        # Learnable token.
        #
        # Initialized with random noise as described
        # in the paper.

        self.token = nn.Parameter(
            torch.randn(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        # Mask only during training

        if not self.training:

            return x

        b, c, h, w = x.shape

        ps = self.patch_size

        # Number of patches

        ph = max(
            1,
            h // ps
        )

        pw = max(
            1,
            w // ps
        )

        total_patches = ph * pw

        number_to_mask = int(
            total_patches * self.mask_ratio
        )

        if number_to_mask <= 0:

            return x

        # Random mask for every image

        mask = torch.zeros(
            b,
            total_patches,
            device=x.device
        )

        for i in range(b):

            indices = torch.randperm(
                total_patches,
                device=x.device
            )[:number_to_mask]

            mask[i, indices] = 1.0

        mask = mask.reshape(
            b,
            ph,
            pw
        )

        # Expand patch mask to image dimensions

        mask = mask.repeat_interleave(
            ps,
            dim=1
        )

        mask = mask.repeat_interleave(
            ps,
            dim=2
        )

        mask = mask[:, :h, :w]

        mask = mask.unsqueeze(1)

        # Replace masked feature regions
        # with learnable token

        token = self.token.expand(
            b,
            -1,
            h,
            w
        )

        x = (
            x * (1.0 - mask)
            +
            token * mask
        )

        return x


# ============================================================
# MCFF
#
# Multichannel Feature Fusion
#
# The paper describes 3 input branches:
#
# 1. current masked feature
# 2. resized feature from another scale
# 3. resized feature from another scale
#
# The fusion is learned by a neural network.
# ============================================================

class MCFF(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            )
        )

    def forward(
        self,
        current,
        branch2,
        branch3
    ):

        # Make sure spatial dimensions match

        if branch2.shape[-2:] != current.shape[-2:]:

            branch2 = F.interpolate(
                branch2,
                size=current.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        if branch3.shape[-2:] != current.shape[-2:]:

            branch3 = F.interpolate(
                branch3,
                size=current.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat(
            [
                current,
                branch2,
                branch3
            ],
            dim=1
        )

        x = self.fusion(x)

        return x


# ============================================================
# MHNet
# ============================================================

class MHNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        mask_ratio=0.25
    ):

        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # ----------------------------------------------------
        # Convolutional embedding
        # ----------------------------------------------------

        self.embedding = ConvEmbedding(
            in_channels,
            c1
        )

        # ----------------------------------------------------
        # Scale 1
        # ----------------------------------------------------

        self.restormer1 = RestormerBlock(
            c1,
            heads=4
        )

        self.mask1 = FeatureMasking(
            c1,
            mask_ratio
        )

        # ----------------------------------------------------
        # Scale 1 -> Scale 2
        #
        # Explicit space-to-depth from paper Eq. (2)
        # ----------------------------------------------------

        self.space_depth1 = SpaceToDepth(2)

        self.down1 = ResizeDown(
            c1 * 4,
            c2
        )

        # ----------------------------------------------------
        # Scale 2
        # ----------------------------------------------------

        self.restormer2 = RestormerBlock(
            c2,
            heads=4
        )

        self.mask2 = FeatureMasking(
            c2,
            mask_ratio
        )

        # ----------------------------------------------------
        # Scale 2 -> Scale 3
        # ----------------------------------------------------

        self.space_depth2 = SpaceToDepth(2)

        self.down2 = ResizeDown(
            c2 * 4,
            c3
        )

        # ----------------------------------------------------
        # Scale 3
        # ----------------------------------------------------

        self.restormer3 = RestormerBlock(
            c3,
            heads=4
        )

        self.mask3 = FeatureMasking(
            c3,
            mask_ratio
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up3 = ResizeUp(
            c3,
            c2
        )

        self.decoder2 = RestormerBlock(
            c2,
            heads=4
        )

        self.up2 = ResizeUp(
            c2,
            c1
        )

        self.decoder1 = RestormerBlock(
            c1,
            heads=4
        )

        # ----------------------------------------------------
        # Multiscale features projected to common channel size
        # ----------------------------------------------------

        self.project2 = nn.Conv2d(
            c2,
            c1,
            kernel_size=1
        )

        self.project3 = nn.Conv2d(
            c3,
            c1,
            kernel_size=1
        )

        # ----------------------------------------------------
        # MCFF
        # ----------------------------------------------------

        self.mcff = MCFF(
            c1
        )

        # ----------------------------------------------------
        # Final Restormer
        # ----------------------------------------------------

        self.final_restormer = RestormerBlock(
            c1,
            heads=4
        )

        # ----------------------------------------------------
        # PixelShuffle
        #
        # Paper Eq. (3):
        #
        # Z = Conv(Ps(Fde))
        # ----------------------------------------------------

        self.pixel_shuffle_conv = nn.Conv2d(
            c1,
            c1 * 4,
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(
            2
        )

        # ----------------------------------------------------
        # Final flood prediction
        #
        # One output channel = flood logit
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            c1,
            1,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual input projection
        #
        # Allows the input image to contribute to final
        # prediction without trying to directly add RGB
        # channels to a single segmentation channel.
        # ----------------------------------------------------

        self.input_projection = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        original = x

        # ====================================================
        # Encoder
        # ====================================================

        # Convolutional embedding

        x1 = self.embedding(x)

        # Restormer + mask

        x1 = self.restormer1(x1)

        x1_masked = self.mask1(x1)

        # ====================================================
        # Scale 2
        # ====================================================

        x2 = self.space_depth1(
            x1_masked
        )

        x2 = self.down1(x2)

        x2 = self.restormer2(x2)

        x2_masked = self.mask2(x2)

        # ====================================================
        # Scale 3
        # ====================================================

        x3 = self.space_depth2(
            x2_masked
        )

        x3 = self.down2(x3)

        x3 = self.restormer3(x3)

        x3_masked = self.mask3(x3)

        # ====================================================
        # Decoder
        # ====================================================

        d2 = self.up3(
            x3_masked
        )

        # Skip connection from scale 2

        if d2.shape[-2:] != x2.shape[-2:]:

            d2 = F.interpolate(
                d2,
                size=x2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = d2 + x2

        d2 = self.decoder2(d2)

        # ----------------------------------------------------

        d1 = self.up2(
            d2
        )

        # Skip connection from scale 1

        if d1.shape[-2:] != x1.shape[-2:]:

            d1 = F.interpolate(
                d1,
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = d1 + x1

        d1 = self.decoder1(d1)

        # ====================================================
        # Multiscale fusion
        # ====================================================

        p2 = self.project2(
            x2_masked
        )

        p3 = self.project3(
            x3_masked
        )

        # Resize to scale-1 resolution

        p2 = F.interpolate(
            p2,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        p3 = F.interpolate(
            p3,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        fused = self.mcff(
            d1,
            p2,
            p3
        )

        # ====================================================
        # Final Restormer
        # ====================================================

        fused = self.final_restormer(
            fused
        )

        # ====================================================
        # PixelShuffle
        # ====================================================

        fused = self.pixel_shuffle_conv(
            fused
        )

        fused = self.pixel_shuffle(
            fused
        )

        # ====================================================
        # Segmentation output
        # ====================================================

        logits = self.output_conv(
            fused
        )

        # ====================================================
        # Residual input
        # ====================================================

        residual = self.input_projection(
            original
        )

        residual = F.interpolate(
            residual,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        logits = logits + residual

        # ====================================================
        # Restore original image resolution
        # ====================================================

        logits = F.interpolate(
            logits,
            size=original.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits