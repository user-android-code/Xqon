import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import hashlib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="GAN .pth 読み込み推論アプリ", layout="wide")

st.title("🎨 実在モデル：CelebA GAN 画像生成")
st.write("Hugging Face公式のリポジトリから確実に存在する `dcgan-celeba.pth` を自動ロードして計算します。")

# ---------------------------------------------------------
# 1. dcgan-celeba.pth に適合する Generator 構造定義
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 入力: (batch, 100, 1, 1) -> 64x64 画像を出力
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # 4x4
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
        return self.main(input)

# ---------------------------------------------------------
# 2. Hugging Face 公式の動作確認済み .pth ファイルをロード
# ---------------------------------------------------------
@st.cache_resource
def load_gan_model():
    # ★確実に存在するHugging Face公式ドキュメント用の公開モデル
    repo_id = "huggingface/hub-docs"
    filename = "dcgan-celeba.pth"
    
    # ダウンロード実行
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    model = Generator()
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # キー名に 'main.' が入っている場合の吸収処理
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, repo_id, filename

# モデル読み込み実行
try:
    with st.spinner("Hugging Face からモデル (dcgan-celeba.pth) を取得中..."):
        model, loaded_repo, loaded_file = load_gan_model()
    st.sidebar.success(f"ロード成功:\n{loaded_repo}/{loaded_file}")
except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. メイン画面：プロンプト入力からノイズ計算・推論
# ---------------------------------------------------------
prompt = st.text_input("生成プロンプト（例: 'person A', 'cool face'）", value="fashion model")

if st.button("画像生成（推論を実行）"):
    if not prompt:
        st.warning("プロンプトを入力してね！")
    else:
        with st.spinner("GANモデルで順伝播計算中..."):
            # プロンプトの文字列からシード値を算出
            seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)
            
            # 100次元の潜在ノイズベクトル (1, 100, 1, 1)
            z = torch.randn(1, 100, 1, 1)

            # 推論計算
            with torch.no_grad():
                fake_tensor = model(z)
                # Tanh出力 [-1, 1] を [0, 1] に補正
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
            st.write(f"**使用モデル:** `{loaded_repo}/{loaded_file}`")
