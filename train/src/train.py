from checkpointer import Checkpointer
import functools
from models import create_discriminator, create_generator
from serialize import deserialize_model, serialize, serialize_model
import tensorflow as tf
from tensor_ops import lerp
from training_loop import training_loop
from typing import Optional


def decode_record_image(record_bytes):
    schema = {
        'image_shape': tf.io.FixedLenFeature([3], dtype=tf.int64),
        'image_bytes': tf.io.FixedLenFeature([], dtype=tf.string)
        }
    example = tf.io.parse_single_example(record_bytes, schema)
    image = tf.io.decode_image(example['image_bytes'])
    image = tf.reshape(image, tf.cast(example['image_shape'], tf.int32))
    return image


def make_real_image_dataset(
        batch_size: int,
        file_pattern: str,
        ) -> tf.data.Dataset:
    file_names = tf.io.gfile.glob(file_pattern)

    return tf.data.TFRecordDataset(file_names
        ).map(decode_record_image
        ).map(lambda image: tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        ).shuffle(1000
        ).repeat(
        ).batch(batch_size
        ).prefetch(tf.data.AUTOTUNE)


class TrainingOptions:
    def __init__(
            self,
            resolution: int,
            replica_batch_size: int,
            real_images_file_pattern: str,
            epoch_sample_count: int = 1024 * 16,
            total_sample_count: int = 1024 * 16 * 32,
            learning_rate: float = 0.002,
            latent_size = 64,
            visualization_smoothing_sample_count: int = 10000,
			beta_1 = None,
			beta_2 = None
            ):
        assert epoch_sample_count % replica_batch_size == 0
        assert total_sample_count % epoch_sample_count == 0

        self.resolution = resolution
        self.replica_batch_size = replica_batch_size
        self.epoch_sample_count = epoch_sample_count
        self.total_sample_count = total_sample_count
        self.learning_rate = learning_rate
        self.real_images_file_pattern = real_images_file_pattern
        self.latent_size = latent_size
        self.visualization_smoothing_sample_count = visualization_smoothing_sample_count
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @property
    def epoch_count(self):
        return self.total_sample_count // self.epoch_sample_count


class TrainingState:
    def __init__(
            self,
            options: TrainingOptions,
            generator: Optional[tf.keras.Model] = None,
            visualization_generator: Optional[tf.keras.Model] = None,
            discriminator: Optional[tf.keras.Model] = None,
            epoch_i: int = 0):
        self.options = options
        self.generator = generator
        self.visualization_generator = visualization_generator
        self.discriminator = discriminator
        self.epoch_i = epoch_i

    def training_is_done(self) -> bool:
        return self.epoch_i * self.options.epoch_sample_count >= self.options.total_sample_count

    def __getstate__(self):
        state = self.__dict__.copy()
        state['generator'] = serialize_model(self.generator)
        state['visualization_generator'] = serialize_model(self.visualization_generator)
        state['discriminator'] = serialize_model(self.discriminator)
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.generator = deserialize_model(
            self.generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))
        self.visualization_generator = deserialize_model(
            self.visualization_generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))
        self.discriminator = deserialize_model(
            self.discriminator,
            functools.partial(create_discriminator, self.options.resolution))


class CheckpointStateCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            state: TrainingState,
            checkpointer: Checkpointer):
        self.state = state
        self.checkpointer = checkpointer
        super().__init__()

    def on_epoch_end(self, epoch_i: int, logs=None) -> None:
        self.state.epoch_i = epoch_i + 1
        self.checkpointer.save_checkpoint(self.state.epoch_i, serialize(self.state))


class WeightDecay(tf.keras.callbacks.Callback):
    def __init__(
            self,
            strategy: tf.distribute.Strategy,
            generator: tf.keras.Model,
            visualization_generator: tf.keras.Model,
            visualization_weight_decay: float):
        self.strategy = strategy
        self.generator = generator
        self.visualization_generator = visualization_generator
        self.visualization_weight_decay = visualization_weight_decay

    @tf.function
    def update_weight_mas(self):
        for ma, v in zip(
                self.visualization_generator.trainable_weights,
                self.generator.trainable_weights):
            ma.assign(lerp(ma, v, 1. - self.visualization_weight_decay))

    def on_train_batch_end(self, batch_i: int, logs=None) -> None:
        self.strategy.run(self.update_weight_mas)


def train(
		strategy: tf.distribute.MirroredStrategy,
		checkpointer: Checkpointer,
		state: TrainingState
		) -> None:
	options = state.options

	# override parameters
	#options.total_sample_count = 4194304

	checkpoint_callback = CheckpointStateCallback(state, checkpointer)

	if state.generator is None:
		global_batch_size = options.replica_batch_size * strategy.num_replicas_in_sync
		with strategy.scope():
			state.generator = create_generator(options.resolution, options.latent_size)
			state.visualization_generator = create_generator(
				options.resolution,
				options.latent_size)
			state.discriminator = create_discriminator(options.resolution)

	global_batch_size = options.replica_batch_size * strategy.num_replicas_in_sync

	visualization_weight_decay = (
		0.5 ** (global_batch_size / options.visualization_smoothing_sample_count)
		if options.visualization_smoothing_sample_count > 0 else
		0.0)
	weight_decay = WeightDecay(
		strategy,
		state.generator,
		state.visualization_generator,
		visualization_weight_decay)

	image_dataset = strategy.experimental_distribute_dataset(
		make_real_image_dataset(
			global_batch_size,
			file_pattern=options.real_images_file_pattern))

	state.epoch_i = training_loop(
		strategy,
		state.generator,
		state.discriminator,
		image_dataset,
		state.epoch_i,
		options.epoch_count,
		options.replica_batch_size,
		options.epoch_sample_count,
		learning_rate=options.learning_rate,
		beta_1=options.beta_1,
		beta_2=options.beta_2,
		callbacks=[weight_decay, checkpoint_callback])
	checkpointer.save_checkpoint(state.epoch_i, serialize(state))













