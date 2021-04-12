import argparse

def get_args_training():
    args = argparse.ArgumentParser(description='Unet')

    args.add_argument('--n_epochs', type=int, default=100, help='The number of epoch')
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--loss_func', type=str, default='structure_loss', help='Loss function')
    args.add_argument('--train_size', type=int, default=256)
    args.add_argument('--test_root', type=str, default='../data/TestDataset')
    args.add_argument('--img_root', type=str, default='../data/TrainDataset/image/')
    args.add_argument('--gt_root', type=str, default='../data/TrainDataset/mask/')
    args.add_argument('--saved_path', type=str, default='./saved_models')
    args.add_argument('--saved_checkpoint', type=str, default='../saved_checkpoint')
    args.add_argument('--logging', type=str, default='./logging')
    args.add_argument('--result_path', type=str, default='./result')
    args.add_argument('--saved_img', type=str, default='../saved_img')
    args.add_argument('--resume', action='store_true', default=False)
    args.add_argument('--backbone', type=str, default='ResEncoder')
    args.add_argument('--bridge', type=str, default='ResBridge')
    args.add_argument('--decoder', type=str, default='ResDecoder')
    args.add_argument('--n_blocks', type=int, default=1)
    
    args = args.parse_args()

    return args