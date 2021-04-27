cd ../../
pwd

# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 256 --batch_size 16
# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 352 --batch_size 16

# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 256 --batch_size 16
# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 352 --batch_size 16



# # Augmentation

# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 256 --batch_size 16 --augmentation
# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 352 --batch_size 16 --augmentation

# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 256 --batch_size 16 --augmentation
# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 352 --batch_size 16 --augmentation


# ### EVAL 
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 256 --batch_size 16
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 352 --batch_size 16

# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 256 --batch_size 16
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 352 --batch_size 16



# # Augmentation

# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 256 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --n_block 1 --train_size 352 --batch_size 16 --augmentation

# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 256 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --n_block 1 --train_size 352 --batch_size 16 --augmentation


#### SimpleDecoder

# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 
# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16


# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16

# ## Augmentation
# python train.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 
# python train.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation


# python eval_model.py --backbone Resnet18 --n_filters 5 --bridge Resnet18 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation 
# python eval_model.py --backbone Resnet34 --n_filters 5 --bridge Resnet34 --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation


