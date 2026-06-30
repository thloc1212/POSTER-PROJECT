# POSTER Reimplementation and Improvement

Repo này triển khai lại [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://github.com/zczcwh/POSTER) dưới dạng các notebook Colab/Jupyter, đồng thời thêm một nhánh cải tiến cho bài toán nhận diện cảm xúc khuôn mặt.

POSTER gốc dùng hai luồng đặc trưng:

- `IR50`: trích đặc trưng ảnh khuôn mặt.
- `MobileFaceNet`: trích đặc trưng landmark khuôn mặt.
- `Pyramid Cross-Fusion Transformer`: fuse hai luồng đặc trưng để phân loại cảm xúc.

Trong repo này, phần chạy chính được gom vào các notebook `.ipynb`, còn script `.py` được dùng làm backend train/test cho từng mô hình.

## Nội dung chính

| File | Vai trò |
| --- | --- |
| `POSTER.ipynb` | Notebook train/test POSTER gốc trên RAF-DB, AffectNet 7, AffectNet 8 và FERPlus. |
| `improvement.ipynb` | Notebook train/test mô hình cải tiến trên RAF-DB, AffectNet 7, AffectNet 8, FERPlus, SFEW và FER2013. |
| `ADDITIONAL_MODEL.ipynb` | Notebook chạy thêm các baseline/ablation như baseline transformer, ResNet18 và POSTER trên nhiều dataset. |
| `demo.ipynb` | Demo so sánh POSTER gốc và mô hình cải tiến, kèm attention/heatmap trực quan. |
| `application.ipynb` | Demo upload ảnh bằng `ipywidgets`, dự đoán cảm xúc và hiển thị confidence/heatmap. |

## Cấu trúc code

```text
.
├── data_preprocessing/          # Dataset loader và tiện ích preprocess
├── models/
│   ├── emotion_hyp.py           # POSTER gốc
│   ├── hyp_crossvit.py          # Pyramid Cross-Fusion Transformer gốc
│   ├── improvement_emotion_hyp.py
│   ├── improvement_hyp_crossvit.py
│   ├── baseline_emotion_hyp.py
│   ├── ir50.py
│   ├── mobilefacenet.py
│   └── pretrain/                # Pretrained IR50/MobileFaceNet
├── train.py / test.py           # Train/test baseline hoặc POSTER gốc
├── improvement_train.py
├── improvement_test.py          # Train/test mô hình cải tiến
├── train_resnet.py
├── test_resnet.py               # Train/test ResNet18 baseline
└── checkpoint/                  # Checkpoint output
```

## Cải tiến so với POSTER

Bản improve đang được dùng trong `improvement_train.py` và `improvement_test.py` nằm ở:

- `models/improvement_emotion_hyp.py`
- `models/improvement_hyp_crossvit.py`

Các thay đổi chính:

1. Thay `Pyramid Cross-Fusion Transformer` bằng `ProductCrossDualAttention`.
   - POSTER gốc gọi `self.pyramid_fuse = HyVisionTransformer(...)`.
   - Bản cải tiến gọi `self.dual_attn_fuse = ProductCrossDualAttention(...)`.
   - Module mới thực hiện cross-attention hai chiều: landmark query sang image key/value và image query sang landmark key/value.

2. Thêm product-cross gating.
   - Hai nhánh cross-attention được concat rồi đưa qua một gate học được.
   - Gate quyết định mức đóng góp giữa image stream và landmark stream.
   - Có `res_scale` là tham số học được để điều tiết residual fusion.

3. Thêm chuẩn hóa sau fusion.
   - Sau khi fuse, đặc trưng đi qua `SE_block`.
   - Sau đó dùng `LayerNorm(512)` trước classifier để ổn định đặc trưng.

4. Thử nghiệm learnable token riêng cho landmark.
   - Trong `models/improvement_hyp_crossvit.py`, `HyVisionTransformer` được sửa để có `lm_cls_token` và `lm_pos_embed` riêng cho landmark stream.
   - Hai class token `image cls` và `landmark cls` được fuse bằng `fuse_proj`.
   - Lưu ý: bản train/test improve hiện tại đang dùng trực tiếp `ProductCrossDualAttention`, không gọi `HyVisionTransformer` cải tiến này. Phần learnable landmark token nên xem như code thử nghiệm/ablation.

Ngoài ra còn có hướng thử nghiệm cũ trong `models/old-improvement_*` và `models/adaptive_fusion_mechanism.py`, nhưng đây không phải nhánh improve đang được script chính sử dụng.

## Cài đặt

Môi trường gốc của POSTER dùng Python 3.9, PyTorch 1.10.2 và CUDA 11.3. Có thể tạo môi trường từ file:

```bash
conda create --name poster --file requirements.txt
conda activate poster
```

Nếu chạy trên Colab, các notebook đã có cell mount Google Drive và tải dataset bằng `kagglehub` ở từng phần tương ứng.

## Pretrained weights

POSTER cần pretrained backbone cho hai nhánh:

```text
models/pretrain/
├── ir50.pth
├── mobilefacenet_model_best.pth.tar
├── HR18-AFLW.pth
└── HR18-WFLW.pth
```

Repo gốc cung cấp pretrained weights tại phần README của [zczcwh/POSTER](https://github.com/zczcwh/POSTER). Đặt toàn bộ file pretrained vào `models/pretrain/`.

## Dataset

Các script hỗ trợ các dataset sau:

| Dataset | Số lớp | Script chính |
| --- | ---: | --- |
| RAF-DB | 7 | `train.py`, `test.py`, `improvement_train.py`, `improvement_test.py` |
| AffectNet 7 | 7 | `train_affect.py`, `test_affect.py`, `improvement_train.py`, `improvement_test.py` |
| AffectNet 8 | 8 | `train_affect.py`, `test_affect.py`, `improvement_train.py`, `improvement_test.py` |
| FERPlus | 8 | `train.py`, `test.py`, `improvement_train.py`, `improvement_test.py` |
| SFEW | 7 | `trainsfew.py`, `testsfew.py`, `improvement_train.py`, `improvement_test.py` |
| FER2013 | 7 | `trainfer2013.py`, `testfer2013.py`, `improvement_train.py`, `improvement_test.py` |

RAF-DB có thể dùng một trong hai cấu trúc:

```text
raf-basic/
├── EmoLabel/
│   └── list_patition_label.txt
└── Image/
    └── aligned/
```

hoặc dạng `ImageFolder`:

```text
DATASET/
├── train/
└── test/
```

SFEW yêu cầu:

```text
sfew/
├── train/
├── val/
└── test/
```

FER2013 yêu cầu:

```text
fer2013/
├── train/
└── test/
```

## Train/test POSTER gốc

Train POSTER trên RAF-DB:

```bash
python train.py --dataset rafdb --datapath "<path-to-rafdb>" --batch_size 64 --modeltype large --epochs 300 --gpu 0 --poster
```

Test POSTER:

```bash
python test.py --dataset rafdb --datapath "<path-to-rafdb>" --checkpoint checkpoint/rafdb_poster_best.pth --gpu 0 -p --poster
```

Nếu không truyền `--poster`, `train.py` và `test.py` sẽ dùng baseline trong `models/baseline_emotion_hyp.py`.

## Train/test mô hình cải tiến

Train bản improve trên RAF-DB:

```bash
python improvement_train.py --dataset rafdb --datapath "<path-to-rafdb>" --batch_size 64 --modeltype large --epochs 300 --gpu 0
```

Test bản improve:

```bash
python improvement_test.py --dataset rafdb --datapath "<path-to-rafdb>" --checkpoint checkpoint/rafdb_improvement_best.pth --gpu 0 -p
```

Train trên AffectNet 7:

```bash
python improvement_train.py --dataset affectnet --datapath "<path-to-affectnet>" --batch_size 64 --modeltype large --epochs 300 --gpu 0
```

Train trên AffectNet 8:

```bash
python improvement_train.py --dataset affectnet8class --datapath "<path-to-affectnet8>" --batch_size 64 --modeltype large --epochs 300 --gpu 0
```

Train trên FERPlus:

```bash
python improvement_train.py --dataset ferplus --datapath "<path-to-ferplus>" --batch_size 64 --modeltype large --epochs 300 --gpu 0
```

Train trên SFEW:

```bash
python improvement_train.py --dataset sfew --datapath "<path-to-sfew>" --batch_size 64 --modeltype large --epochs 300 --gpu 0
```

Train trên FER2013:

```bash
python improvement_train.py --dataset fer2013 --datapath "<path-to-fer2013>" --batch_size 128 --modeltype large --epochs 300 --gpu 0
```

Checkpoint mặc định được lưu vào:

```text
checkpoint/<dataset>_improvement_latest.pth
checkpoint/<dataset>_improvement_best.pth
```

Có thể đổi nơi lưu bằng:

```bash
--save_dir checkpoint-save/additional_model --save_prefix my_run
```

## ResNet18 baseline

Train ResNet18:

```bash
python train_resnet.py --dataset rafdb --datapath "<path-to-rafdb>" --batch_size 64 --epochs 20 --gpu 0
```

Test ResNet18:

```bash
python test_resnet.py --dataset rafdb --datapath "<path-to-rafdb>" --checkpoint checkpoint/rafdb_resnet18_best.pth --gpu 0 -p
```

## Notebook workflow

Khuyến nghị chạy theo thứ tự:

1. `POSTER.ipynb`: kiểm tra lại model POSTER gốc.
2. `improvement.ipynb`: train/test bản cải tiến.
3. `ADDITIONAL_MODEL.ipynb`: chạy baseline/ResNet để đối chiếu.
4. `demo.ipynb` hoặc `application.ipynb`: upload ảnh và so sánh dự đoán trực quan.

Trong notebook, thay các biến `path` hoặc `--datapath` thành đường dẫn dataset thực tế trên máy/Google Drive.

## Citation

Nếu dùng lại POSTER gốc, trích dẫn paper:

```bibtex
@article{zheng2022poster,
  title={Poster: A pyramid cross-fusion transformer network for facial expression recognition},
  author={Zheng, Ce and Mendieta, Matias and Chen, Chen},
  journal={arXiv preprint arXiv:2204.04083},
  year={2022}
}
```

## License

Code kế thừa từ POSTER gốc và giữ theo giấy phép trong `LICENSE`.
