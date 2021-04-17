cd ../../
pwd


python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352
python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352
python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352

