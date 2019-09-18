#!/bin/sh

# Download the MNIST image data and corresponding labels

URL='http://yann.lecun.com/exdb/mnist'

TRAIN_IMAGES='train-images-idx3-ubyte.gz'
TRAIN_LABELS='train-labels-idx1-ubyte.gz'
TEST_IMAGES='t10k-images-idx3-ubyte.gz'
TEST_LABELS='t10k-labels-idx1-ubyte.gz'

MNIST_DATA_DIR='data/mnist'


if [ ! -e $MNIST_DATA_DIR ]; then
    echo "Create ${MNIST_DATA_DIR}"
    mkdir -p $MNIST_DATA_DIR
fi
for file in $TRAIN_IMAGES $TRAIN_LABELS $TEST_IMAGES $TEST_LABELS
do
    if [ -e $MNIST_DATA_DIR/$file ]; then
        echo "${file} already downloaded"
    else
        echo "Downloading ${URL}/${file}..."
        wget -P $MNIST_DATA_DIR $URL/$file
    fi
done
