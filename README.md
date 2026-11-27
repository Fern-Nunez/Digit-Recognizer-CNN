# Digit Recognizer (CNN)

A convolutional neural network that classifies handwritten digits (0 to 9) from the MNIST dataset, built in Python with TensorFlow and Keras.

Course project for [CSE 5160 Machine Learning / course name], California State University, San Bernardino (2023).

## What it does

- Loads the MNIST dataset (60,000 training images, 10,000 test images of 28x28 grayscale digits)
- Preprocesses the data: reshapes images to 28x28x1, normalizes pixel values from 0-255 to 0-1, and one-hot encodes the labels
- Trains a CNN and saves it as `digitRecognizer.h5`
- Loads the saved model, evaluates it on the test set, and predicts the digit in a new image

## Model architecture

| Layer | Details |
|---|---|
| Conv2D | 30 filters, 5x5 kernel, ReLU |
| MaxPooling2D | 2x2, stride 2 |
| Conv2D | 15 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2, stride 2 |
| Dropout | 20% |
| Flatten | |
| Dense | 128 units, ReLU |
| Dense | 50 units, ReLU |
| Dense (output) | 10 units, softmax |

- Loss: categorical cross-entropy
- Optimizer: Adam
- Training: 10 epochs, batch size 200

## Results

Test accuracy: **[X]%** on the 10,000-image MNIST test set.

## Files

- `digitRecognizer.py`: loads and preprocesses the data, builds and trains the model, saves it to `digitRecognizer.h5`, and predicts a sample image
- `[second_file_name].py`: loads the saved model, evaluates it on the test set, and predicts the digit in a new image

## How to run

```bash
pip install tensorflow
python digitRecognizer.py
python [second_file_name].py
```

To test your own image, change the image path in `test_model_performance()` to a 28x28 grayscale image of a single digit.

## Note

Written in 2023. Newer Keras versions removed `keras.utils.np_utils`. If you get an import error, replace `np_utils.to_categorical` with `tf.keras.utils.to_categorical`.
