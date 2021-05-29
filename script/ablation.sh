cd ../
pwd


################### ABLATION ENCODER

# python train.py --backbone ResEncoder --n_filters 4 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  
# # python train.py --backbone Resnet18 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# # python train.py --backbone Resnet34 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python train.py --backbone Resnet50 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation    #79.2M
# python train.py --backbone res2net50_26w_4s --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 79.3M
# python train.py --backbone res2net50_48w_2s --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 79.1M



# python eval_model.py --backbone ResEncoder --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# # python eval_model.py --backbone Resnet18 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# # python eval_model.py --backbone Resnet34 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone Resnet50 --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone res2net50_26w_4s --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
# python eval_model.py --backbone res2net50_48w_2s --n_filters 5 --bridge ResBridge --decoder SimpleDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation




################### ABLATION BRIDGE
python train.py --backbone Resnet34 --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation    # 14.23M
python train.py --backbone Resnet50 --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation    # 37.94M
python train.py --backbone res2net50_26w_4s --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 38.05M
python train.py --backbone res2net50_48w_2s --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 37.89M


python eval_model.py --backbone Resnet34 --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation    #79.2M
python eval_model.py --backbone Resnet50 --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation    #79.2M
python eval_model.py --backbone res2net50_26w_4s --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 79.3M
python eval_model.py --backbone res2net50_48w_2s --n_filters 5 --bridge ResBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation  # 79.1M














################### ABLATION DECODER    