cd ../
pwd


python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation

python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 

python train.py --backbone Resnet50 --n_filters 5 --bridge Resnet50 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation


python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 

python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 

python eval_model.py --backbone Resnet50 --n_filters 5 --bridge Resnet50 --decoder DepthwsDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 
