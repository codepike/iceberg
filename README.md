# iceberg
This  is a project of iceberg.

Analyze data
$ipython notebook analyze_data.ipynb

Download train data and sample data to ./data

$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/final.tfrecord --start 90 --end 100

Train a model
$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/train.tfrecord --start 0 --end 84.415 --flip --rotate 8 --flip_rotate 8
$python iceberg/main.py  --mode train --epoch 10000 --batch_size 128 --logdir logdir  --data_path ./data/train.tfrecord

Evaluate a model
$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/validation.tfrecord --start 70 --end 90
$python iceberg/main.py  --mode evaluate --epoch 1 --batch_size 128 --logdir logdir  --data_path ./data/validation.tfrecord

Predict the result
Create a test tfrecord file
$python iceberg/create_dataset.py --data_path ./data/test.json --output ./data/test.tfrecord --start 0 --end 100
$python iceberg/main.py --data_path ./data/test.tfrecord --epoch 1 --batch_size 128 --output test.csv --logdir logdir --mode batch_predict
$python prepare_submission.py