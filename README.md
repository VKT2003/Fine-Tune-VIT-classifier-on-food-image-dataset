# Fine-Tuning ViT Classifier on Food Image Dataset

This repository provides the code and guidelines for fine-tuning the Vision Transformer (ViT) model on a food image dataset. The goal is to improve the model's performance in classifying different types of food images.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Dataset](#dataset)
- [Fine-Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Vision Transformer (ViT) model leverages the transformer architecture for image classification tasks, offering a powerful alternative to convolutional neural networks. This repository demonstrates how to fine-tune a pre-trained ViT model on a food image dataset to enhance its classification capabilities.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch 1.8.1 or higher
- Transformers library by Hugging Face
- torchvision
- CUDA (if using a GPU for training)

## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/vit-food-classifier.git
    cd vit-food-classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Prepare the food image dataset for training. You can use a publicly available food image dataset or your custom dataset. Ensure the dataset is organized in the following structure:
```
data/
  train/
    class1/
      img1.jpg
      img2.jpg
      ...
    class2/
      img1.jpg
      img2.jpg
      ...
    ...
  val/
    class1/
      img1.jpg
      img2.jpg
      ...
    class2/
      img1.jpg
      img2.jpg
      ...
    ...
```

## Fine-Tuning

To fine-tune the ViT model on the food image dataset, execute the following command:

```bash
python fine_tune.py --dataset data --model vit --output_dir models/vit-food
```

### Fine-Tuning Parameters

- `--dataset`: Path to the dataset directory.
- `--model`: Name or path of the pre-trained ViT model.
- `--output_dir`: Directory where the fine-tuned model will be saved.

Additional training parameters such as batch size, learning rate, and number of epochs can be customized in the `fine_tune.py` script.

## Evaluation

After fine-tuning, evaluate the model to verify its performance. Run the evaluation script as follows:

```bash
python evaluate.py --model models/vit-food --dataset data
```

This script will compute and display performance metrics such as accuracy, precision, recall, and F1 score.

## Results

The results of the fine-tuning process, including training and evaluation metrics, will be saved in the `results/` directory. Key metrics to consider are accuracy, precision, recall, and F1 score.

## Usage

To use the fine-tuned ViT model for inference, load it using the Transformers library:

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

feature_extractor = ViTFeatureExtractor.from_pretrained("models/vit-food")
model = ViTForImageClassification.from_pretrained("models/vit-food")

image = Image.open("path_to_your_image.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()

print(f"Predicted class: {predicted_class}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

Thank you for your interest in this project! If you have any questions or feedback, please open an issue or contact the repository maintainer.
