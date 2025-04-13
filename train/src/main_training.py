import argparse
from typing import List, Optional
from checkpointer import Checkpointer
import os
from serialize import deserialize
import tensorflow as tf
import time
from train import TrainingOptions, TrainingState, train

def init_training(
        args: argparse.Namespace):

    strategy = tf.distribute.MirroredStrategy()

    checkpointer = Checkpointer(args.out)
    training_options = TrainingOptions(
        args.resolution,
        args.replica_batch_size,
        args.dataset_file_pattern,
        epoch_sample_count=args.epoch_sample_count,
        total_sample_count=args.total_sample_count,
        learning_rate=args.learning_rate,
        latent_size=args.latent_size,
        beta_1=args.beta_1,
        beta_2=args.beta_2)
    training_state = TrainingState(training_options)

    train(
        strategy,
        checkpointer,
        training_state)


def resume_training(
        args: argparse.Namespace):
	strategy = tf.distribute.MirroredStrategy()
	checkpointer = Checkpointer(args.out)

	with strategy.scope():
		training_state = deserialize(checkpointer.load_checkpoint())

	train(
		strategy,
		checkpointer,
		training_state)


def main(
        raw_arguments: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description='Train a GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser(
        'init',
        help='initialize and begin training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    init_parser.set_defaults(func=init_training)

    init_parser.add_argument(
        '--resolution',
        type=int,
        help='resolution of generated images. Must match the resolution of the dataset',
        default=512)

    init_parser.add_argument(
        '--replica-batch-size',
        type=int,
        help='size of batch per replica',
        default=8)

    init_parser.add_argument(
        '--epoch-sample-count',
        type=int,
        help='number of samples per epoch. Must divide evenly into total-sample-count',
        default=16*1024)

    init_parser.add_argument(
        '--total-sample-count',
        type=int,
        help='number of total samples to train on. Must be divisible by epoch-sample-count',
        default=25*1024*1024)

    init_parser.add_argument(
        '--learning-rate',
        type=float,
        help='learning rate of training loop',
        default=0.002)

    init_parser.add_argument(
        '--latent-size',
        type=int,
        help='size of the generator\'s latent vector',
        default=512)

    init_parser.add_argument(
        '--visualization-smoothing-sample-count',
        type=float,
        help='the factor by which to decay the visualization weights. If 0, no smoothing will be applied.',
        default=10000)

    init_parser.add_argument(
        '--beta-1',
        type=float,
        help='beta_1 for Adam optimizer',
        default=0.0)

    init_parser.add_argument(
        '--beta-2',
        type=float,
        help='beta_2 for Adam optimizer',
        default=0.99)

    init_parser.add_argument(
        'dataset_file_pattern',
        help='GLOB pattern for the dataset files. Ex: \'D:/datasets/ffhq/1024x1024/*.tfrecord\'')

    resume_parser = subparsers.add_parser('resume', help='resume training from a checkpoint')
    resume_parser.set_defaults(func=resume_training)

    for subparser in [init_parser, resume_parser]:
        subparser.add_argument('out', help='root output folder')

    args = parser.parse_args(args=raw_arguments)

    args.func(args)


if __name__ == '__main__':
    main()


