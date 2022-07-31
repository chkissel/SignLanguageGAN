# -*- coding: utf-8 -*-
import argparse
from src.autoencoder import * 
from src.pix2pix import * 
from src.sgnlgan import * 
from src.test import * 
from src.human_semantic_parser import * 
from src.video_generator import * 
from src.lambda_study import * 
from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default='sgnlgan', help="which network to use")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
parser.add_argument("--loader", type=str, default="DataReader", help="Dataloader")
parser.add_argument("--images_dir", type=str, default="", help="images directory")
parser.add_argument("--conditions_dir", type=str, default="", help="conditional images directory")
parser.add_argument("--targets_dir", type=str, default="", help="target images directory")
parser.add_argument("--dataset_name", type=str, default="autoencoder", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--loss", type=str, default="MSE", help="used loss-function")
parser.add_argument(
    "--sample_interval", type=int, default=350, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--dynamic_lambda", type=bool, default=False, help="activate dynamic lambda weightening")
parser.add_argument("--lambda_low", type=int, default=50, help="lower bound for dynamic lambda")
parser.add_argument("--lambda_high", type=int, default=150, help="Upper bound for dynamic lambda")
parser.add_argument("--batch_interval", type=int, default=100, help="interval for ssim checks")
config = parser.parse_args()

def main(config):
    """
    Args:
        config: Parsed arguments from command line.
    """
    start = timer()
    network = None 
    print(config)

    if config.network == 'autoencoder':
        autoencoder = AutoEncoder(config, 'conditional')
        autoencoder.train()

    elif config.network == 'pix2pix':
        gan = Pix2Pix(config, 'conditional')
        gan.train()

    elif config.network == 'human_semantic_parser':
        parser = HumanSemanticParser(config, 'parser')
        parser.train()

    elif config.network == 'sgnlgan':
        pipeline = SgnlGAN(config)
        pipeline.train()
    
    elif config.network == 'video':
        video = VideoGenerator(config)
        video.generate()
    
    elif config.network == 'lambda_study':
        study = LambdaStudy(config)
        study.train()

    else:
        print(f"Unknown network: {config.network}")
        network = 0

    end = timer()
    if network != 0:
        print(f'The SignLanguageGAN has finished.\nTotal processing time for {config.n_epochs - config.epoch} epochs: {end - start} seconds')
if __name__ == "__main__":
    main(config)

