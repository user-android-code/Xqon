import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import hashlib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="ポケモンGAN推論", layout="wide")

st.title("👾 実在モデル：ポケモンGAN画像生成")
st.write("Hugging Faceに実在する `rmachado23/pokemon-gan` から重み (.pth) を自動取得して画像を計算生成します。")

# ---------------------------------------------------------
# 1. rmachado23/pokemon-gan の Generator 構造定義
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 入力: (batch, 100, 1, 1) または (batch, 100) -> 64x64へデコンボリューション
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False), # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # 64x64
            nn.Tanh()
        )

    def forward(self, input):
        # 1次元ベクトルで入ってきた場合、(batch, 100, 1, 1)に整形
        if input.dim() == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)
        return self.main(input)

# ---------------------------------------------------------
# 2. 実在する Hugging Face モデルのロード
# ---------------------------------------------------------
@st.cache_resource
def load_gan_model():
    # ★実在するリポジトリとファイル名
    repo_id = "rmachado23/pokemon-gan"
    filename = "pokemon_gan_generator.pth"
    
    # Hugging Faceから自動ダウンロード
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    model = Generator()
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model, repo_id, filename

# モデル取得実行
try:
    with st.spinner("実在モデル (rmachado23/pokemon-gan) を取得中..."):
        model, loaded_repo, loaded_file = load_gan_model()
    st.sidebar.success(f"ロード完了:\n{loaded_repo}/{loaded_file}")
except Exception as e:
    st.error(f"ダウンロードまたはモデル読み込みエラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. メイン画面：プロンプト入力から計算・生成
# ---------------------------------------------------------
prompt = st.text_input("生成プロンプト（例: 'pikachu', 'fire dragon'）", value="electric rodent")

if st.button("生成計算を実行"):
    if not prompt:
        st.warning("プロンプトを入力してね！")
    else:
        with st.spinner("GANモデルで画像計算中..."):
            # プロンプトの文字列をシード（ハッシュ化）に変換
            seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)
            
            # 100次元のノイズベクトルを作成 (1, 100, 1, 1)
            z = torch.randn(1, 100, 1, 1)

            # 推論計算（順伝播）
            with torch.no_grad():
                fake_tensor = model(z)
                # Tanhの出力 [-1, 1] を [0, 1] に変換
                fake_tensor = (fake_tensor + 1) / 2.0
                
                # Pillow画像に変換
                grid = vutils.make_grid(fake_tensor, normalize=False)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                result_img = Image.fromarray(ndarr)

        # 結果表示
        col1, col2 = st.columns(2)
        with col1:
            st.image(result_img, caption=f"生成結果: '{prompt}'", width=256)
        with col2:
            st.subheader("計算ステータス")
            st.write(f"**生成シード値:** `{seed}`")
            st.write(f"**モデル:** `{loaded_repo}/{loaded_file}`")
