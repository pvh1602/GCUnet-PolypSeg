cd ../../
pwd


# python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352
# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --resume # running at epoch 9 
# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352





# python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16 
# python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16
# python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16


# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16
# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16
# python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16


# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16
# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16 --resume
# python train.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16



# Augmentation 
python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16 --augmentation 
python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16 --augmentation


python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16 --augmentation





# DOne all above

# Eval

# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352


# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352

# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 256
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 256
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 256


# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 256
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 256
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 256


# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 256
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 256
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 256





# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16 
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16


# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 2 --train_size 352 --batch_size 16
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge MHSABridge --n_block 3 --train_size 352 --batch_size 16


# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge MHSABridge --n_block 1 --train_size 352 --batch_size 16