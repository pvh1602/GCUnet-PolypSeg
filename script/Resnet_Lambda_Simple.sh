cd ..
pwd

# python train.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python train.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
# python train.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


# python train.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python train.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
# python train.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


# python train.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation --r 5
# python train.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation --r 5
# python train.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation --r 5






# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


python eval_model.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation --r 5
python eval_model.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation --r 5
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge LambdaBridge --decoder SimpleDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation --r 5
