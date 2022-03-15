# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make a dataset by resizing and subsampling a tfds image dataset."""
from typing import Iterator

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("prefix", None, "Prefix of tfrecord / the split name of"
                    " the dataset.")
flags.DEFINE_string("data_dir", None, "The output dataset directory.")
flags.DEFINE_integer("image_size", 64, "Size of image to resize to.")
flags.DEFINE_integer("num_shards", None, "Number of shards to write out.")
flags.DEFINE_string("dataset", "imagenet2012", "Name of tfds dataset.")
flags.DEFINE_string("split", "train", "Split of data to read from tfds.")
flags.DEFINE_integer("start_perc", 0,
                     "Start percent of data to use. Between 0 and 100")
flags.DEFINE_integer("stop_perc", 100,
                     "Stop percent of data to use. Between 0 and 100")
flags.DEFINE_boolean("local_test", False,
                     "Test functions locally instead of with beam.")

# Feature descriptions need to be specified upfront when writing a new dataset
# via beam.
_NUM_SPLIT = 100


def iterator_from_perc(idx: int) -> Iterator[tf.train.Example]:
  """Iterator yielding from the idx'th percent to idx+1'th percent.

  This is used to let beam load a tfds dataset from different workers at the
  same time!

  Args:
    idx: index between 0 and 100 to load.

  Yields:
    Resized examples starting from the idx'th percent to the idx+1 percent.
  """
  offset = (FLAGS.stop_perc - FLAGS.start_perc) / 100.
  from_ = FLAGS.start_perc + offset * idx
  to = FLAGS.start_perc + offset * (idx + 1)
  logging.info("Building: %d-%d", FLAGS.start_perc, FLAGS.stop_perc)
  logging.info("On slice: %d-%d", from_, to)

  ri = tfds.core.ReadInstruction(FLAGS.split, from_=from_, to=to, unit="%")
  dataset = tfds.load(FLAGS.dataset, split=ri)

  image_size = (FLAGS.image_size, FLAGS.image_size)

  def preproc(dd):
    if tuple(dd["image"].shape[0:2]) != image_size:
      dd["image"] = tf.image.resize(dd["image"], image_size)

    dd["image"] = tf.cast(dd["image"], tf.uint8)
    dd["label"] = tf.cast(dd["label"], tf.int32)
    dd["image"].set_shape(list(image_size) + [3])
    return {"image": dd["image"], "label": dd["label"]}

  dataset = dataset.map(preproc)

  def to_bytes_list(value):
    return [value.tobytes()]

  for flat_record in tfds.as_numpy(dataset):
    features = {}
    for k, v in flat_record.items():
      features[k] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=to_bytes_list(v)))
    yield tf.train.Example(features=tf.train.Features(feature=features))


def pipeline(root):
  output_path = f"{FLAGS.data_dir}/{FLAGS.dataset}_{FLAGS.image_size}/{FLAGS.prefix}.tfrecords"

  logging.info(f"Writing file: {output_path} with prefix: {FLAGS.prefix}")  # pylint: disable=logging-fstring-interpolation

  compression_type = beam.io.filesystem.CompressionTypes.GZIP

  coder = beam.coders.ProtoCoder(tf.train.Example)

  return (root
          | "ReadFromRange" >> beam.Create(range(_NUM_SPLIT))
          | "GenerateRow" >> beam.FlatMap(iterator_from_perc)
          | "WriteToTFRecord" >> beam.io.WriteToTFRecord(
              output_path,
              coder=coder,
              compression_type=compression_type,
              num_shards=FLAGS.num_shards))


def main(unused_argv):

  if FLAGS.local_test:
    for _ in tqdm.tqdm(iterator_from_perc(0)):
      pass
    return


  raise NotImplementedError("OSS Beam pipeline is not yet working!")


if __name__ == "__main__":
  flags.mark_flag_as_required("prefix")
  app.run(main)
