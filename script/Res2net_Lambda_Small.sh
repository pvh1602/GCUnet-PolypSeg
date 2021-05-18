cd ../
pwd 





python train.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


python train.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
python train.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation




python eval_model.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
python eval_model.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
python eval_model.py --backbone res2net50_48w_2s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation


python eval_model.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 1 --train_size 352 --batch_size 16 --augmentation
python eval_model.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 2 --train_size 352 --batch_size 16 --augmentation
python eval_model.py --backbone res2net50_26w_4s --n_filters 5 --bridge LambdaBridge --decoder SmallDecoder --n_block 3 --train_size 352 --batch_size 16 --augmentation

