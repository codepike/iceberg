# iceberg
This  is a project of iceberg.

Analyze data
$ipython notebook analyze_data.ipynb

Download train data and sample data to ./data
$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/train.tfrecord --start 0 --end 70
$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/validation.tfrecord --start 70 --end 90
$python iceberg/create_dataset.py --data_path ./data/train.json --output ./data/final.tfrecord --start 90 --end 100



Train a model
$python iceberg/main.py  --mode train --epoch 1 --batch_size 256 --logdir logdir  --data_path ./data/train.tfrecord

Evaluate a model
$python iceberg/main.py  --mode evaluate --epoch 1 --batch_size 64 --logdir logdir  --data_path ./data/validation.tfrecord

Predict the result
$python iceberg/main.py  --mode predict --logdir logdir  --data_path ./data/test.json