"""fingers dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
from tensorflow import keras

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for fingers dataset."""

  VERSION = tfds.core.Version('1.3.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.1.0': 'Load images as arrays instead of providing the PNG data.',
      '1.2.0': 'Make images grayscale.',
      '1.3.0': 'Switch back to providing paths.'
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(fingers): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 1)),
            'label': tfds.features.ClassLabel(names=['0','1', '2','3','4','5']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(Path('/kaggle/input/fingers/train')),
        'test': self._generate_examples(Path('/kaggle/input/fingers/test'))
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(fingers): Yields (key, example) tuples from the dataset
    for f in path.glob('*.png'):
      # get the label
      end = f.stem.split("_")[1]
      # currently we're focusing on number detection, so just pass the number
      label = end[0]
      yield 'key', {
          'image': f,
          'label': label,
      }
