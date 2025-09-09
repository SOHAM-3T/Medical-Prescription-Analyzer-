# 🩺 Medical Prescription Analyzer

An end-to-end **OCR pipeline for medical prescriptions**, powered by a **fine-tuned Donut (Document Understanding Transformer)** model.  
This project extracts text from handwritten/typed prescriptions and makes it available for further analysis (e.g., medicine recognition, dosage extraction).

---

## 🚀 Features

- 📄 **OCR with Donut** — Fine-tuned Donut model on prescription dataset
- 🧩 **Pipeline Integration** — Easily plug into existing medical text-processing workflows
- 📊 **Evaluation** — Tracks WER (Word Error Rate) and training loss
- 🔧 **Customizable** — Extendable to structured parsing (medicine, dosage, duration)
- 💾 **Reusable Model** — Saved and reloadable with Hugging Face Transformers

---

## 📂 Project Structure

```
Medical-Prescription-Analyzer-/
├── model_fine_tuning/          # Kaggle notebook for fine-tuning Donut
│   └── fine_tuning_model_gamma.ipynb
├── pipeline/                   # Pipeline notebook for inference & analysis
│   └── prescription_pipeline.ipynb
├── README.md                   # Project overview
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── .gitignore                  # Ignored files
```

---

## ⚙️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/SOHAM-3T/Medical-Prescription-Analyzer-.git
   cd Medical-Prescription-Analyzer-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Fine-Tuning (Optional)

To fine-tune Donut on your own dataset:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained Donut
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Fine-tuning is implemented in model_fine_tuning/fine_tuning_model_gamma.ipynb
```

---

## 🔍 Inference Pipeline

Example usage in your pipeline:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

MODEL_DIR = "path/to/donut-finetuned-prescription-ocr"

# Load fine-tuned model
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

# OCR function
def ocr_prescription(img_path: str):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)
    outputs = model.generate(pixel_values, max_length=512)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(ocr_prescription("sample_prescription.jpg"))
```

---

## 📊 Results

| Metric | Value (10 Epochs) |
|--------|-------------------|
| Training Loss | ~4.4 |
| Validation Loss | ~4.4 |
| WER | ~0.87 → 1.15 (small dataset, unstable) |

> **Note:** With a larger dataset and more training epochs, performance will improve significantly.

---

## 📦 Model Access

The fine-tuned model is available in:
- **Kaggle notebook outputs:** `soham3ripathy/fine-tuning-model-gamma`
- Can be uploaded as a Kaggle dataset or GitHub release

---

## 🔮 Future Work

- [ ] Expand dataset for better generalization
- [ ] Add structured JSON parsing (medicine, dosage, duration)
- [ ] Deploy as an API (Flask/FastAPI)
- [ ] Improve model performance with more training data
- [ ] Add support for multiple languages

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to improve.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributors

Thanks to these amazing people who have contributed to this project:

<!-- Add contributors here -->
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/SOHAM-3T">
        <img src="https://github.com/SOHAM-3T.png" width="100px;" alt="SOHAM-3T"/>
        <br />
        <sub><b>Soham Tripathy</b></sub>
      </a>
      <br />
    </td>
      <td align="center">
      <a href="https://github.com/Ajay-Kumar-Prasad">
        <img src="https://github.com/Ajay-Kumar-Prasad.png" width="100px;" alt="SOHAM-3T"/>
        <br />
        <sub><b>Ajay Kumar Prasad</b></sub>
      </a>
      <br />
    </td>
    <!-- Add more contributors as needed -->
      <td align="center">
      <a href="https://github.com/SaiPrithvi1278">
        <img src="https://github.com/SaiPrithvi1278.png" width="100px;" alt="SOHAM-3T"/>
        <br />
        <sub><b>Sai Prithvi</b></sub>
      </a>
      <br />
    </td>
  </tr>
</table>

---

## 🙏 Acknowledgments

- [Naver Clova IX](https://github.com/clovaai/donut) for the original Donut model
- Hugging Face for the Transformers library
- The medical community for inspiring this healthcare technology solution
