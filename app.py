import torch
import torch.nn as nn

# --- Generator (生成器) ---
class Generator256CPU(nn.Module):
    def __init__(self, z_dim=100, feature_maps=32):
        """
        CPU軽量版 Generator (256x256)
        計算量を抑えるため feature_maps の初期値を小さめ(32)に設定
        """
        super().__init__()
        
        # 潜在ベクトル z (100) -> 4x4 に投影
        self.fc = nn.Sequential(
            nn.Linear(z_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.feature_maps = feature_maps

        # アップサンプリングブロック (4x4 -> 256x256)
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            )

        self.net = nn.Sequential(
            # 4x4 -> 8x8
            up_block(feature_maps * 8, feature_maps * 8),
            # 8x8 -> 16x16
            up_block(feature_maps * 8, feature_maps * 4),
            # 16x16 -> 32x32
            up_block(feature_maps * 4, feature_maps * 4),
            # 32x32 -> 64x64
            up_block(feature_maps * 4, feature_maps * 2),
            # 64x64 -> 128x128
            up_block(feature_maps * 2, feature_maps),
            # 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # 出力を [-1, 1] に正規化
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.feature_maps * 8, 4, 4)
        return self.net(x)

# --- Discriminator (識別器) ---
class Discriminator256CPU(nn.Module):
    def __init__(self, feature_maps=32):
        """
        CPU軽量版 Discriminator (256x256)
        """
        super().__init__()
        
        def down_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            # 256x256 -> 128x128
            down_block(3, feature_maps, normalize=False),
            # 128x128 -> 64x64
            down_block(feature_maps, feature_maps * 2),
            # 64x64 -> 32x32
            down_block(feature_maps * 2, feature_maps * 4),
            # 32x32 -> 16x16
            down_block(feature_maps * 4, feature_maps * 4),
            # 16x16 -> 8x8
            down_block(feature_maps * 4, feature_maps * 8),
            # 8x8 -> 4x4
            down_block(feature_maps * 8, feature_maps * 8),
            # 判定 (1次元へ)
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)

# --- 動作確認用メイン処理 ---
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"実行デバイス: {device}")

    # モデルインスタンス化
    netG = Generator256CPU().to(device)
    netD = Discriminator256CPU().to(device)

    # 1. 画像生成のテスト (推論)
    z = torch.randn(2, 100, device=device) # バッチサイズ2, 潜在変数100
    fake_images = netG(z)
    print(f"生成画像サイズ: {fake_images.shape}") # [2, 3, 256, 256]

    # 2. 識別器のテスト
    validity = netD(fake_images)
    print(f"識別結果サイズ: {validity.shape}") # [2, 1]
