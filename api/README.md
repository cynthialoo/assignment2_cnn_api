# Assignment 2: CNN Image Classifier API

This project extends the Assignment 1 FastAPI application with a PyTorch
Convolutional Neural Network image classifier for CIFAR-10.

The original embedding endpoints are still available. The new Assignment 2
endpoint is:

```text
POST /classify-image
```

It accepts an uploaded image and returns the predicted CIFAR-10 class.

## CNN Architecture

The model follows the Assignment 2 specification:

```text
Input: RGB image, 64 x 64 x 3
Conv2D: 16 filters, 3 x 3 kernel, stride 1, padding 1
ReLU
MaxPool2D: 2 x 2 kernel, stride 2
Conv2D: 32 filters, 3 x 3 kernel, stride 1, padding 1
ReLU
MaxPool2D: 2 x 2 kernel, stride 2
Flatten
Fully connected: 100 units
ReLU
Fully connected: 10 units
```

The spatial dimensions flow as:

```text
64 x 64 x 3
64 x 64 x 16
32 x 32 x 16
32 x 32 x 32
16 x 16 x 32
8192
100
10
```

## Train the Model

From this directory:

```bash
python train_cnn.py --epochs 10 --batch-size 64
```

The trained checkpoint is saved to:

```text
models/cifar10_cnn.pth
```

For a quick smoke test, use a smaller subset:

```bash
python train_cnn.py --epochs 1 --limit-train 512 --limit-test 256
```

## Run with Docker

Build the image:

```bash
docker build -t assignment2-cnn-api .
```

Run the API:

```bash
docker run --rm -p 8000:8000 assignment2-cnn-api
```

Open the docs:

```text
http://localhost:8000/docs
```

## Test the CNN Endpoint

Upload an image:

```bash
curl -X POST "http://localhost:8000/classify-image" \
  -F "file=@/path/to/image.png"
```

The response includes:

```json
{
  "predicted_class": "frog",
  "predicted_index": 6,
  "confidence": 0.21,
  "model_loaded": true,
  "probabilities": {
    "airplane": 0.01,
    "automobile": 0.02
  }
}
```

The exact prediction and confidence can change depending on the uploaded image
and how long the model was trained. For this assignment, I trained the model
mainly to demonstrate the CNN architecture and the API inference workflow, so a
short training run may not classify every image correctly.

If `models/cifar10_cnn.pth` has not been created yet, the API still starts but
returns `model_loaded: false` with a warning. Train the model before final
submission so inference uses learned weights.

## Other Endpoints

The Assignment 1 endpoints remain available:

```text
GET /embedding?word=apple
GET /similarity?word1=apple&word2=car
GET /cifar10/classes
GET /health
```
