#!/usr/bin/env python3

import xxhash
import struct
import gzip
import sys
import os
import re
import io
import cv2
import numpy as np
import tensorflow as tf
import pkgutil
import google.protobuf.message
from tqdm import tqdm
from google.protobuf.json_format import MessageToDict


# Open up our message output directory to get our protobuf types
shared_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "shared")

sys.path.append(shared_path)

# Load all of the protobuf files as modules to use
for dir_name, subdir, files in os.walk(shared_path):
    modules = pkgutil.iter_modules(path=[dir_name])
    for loader, module_name, ispkg in modules:
        if module_name.endswith("pb2"):

            # Load our protobuf module
            module = loader.find_module(module_name).load_module(module_name)

parsers = {}

# Now that we've imported them all get all the subclasses of protobuf message
for message in google.protobuf.message.Message.__subclasses__():

    # Work out our original protobuf type
    pb_type = message.DESCRIPTOR.full_name.split(".")[1:]
    pb_type = ".".join(pb_type)

    # Reverse to little endian
    pb_hash = bytes(reversed(xxhash.xxh64(pb_type, seed=0x4E55436C).digest()))

    parsers[pb_hash] = (pb_type, message)


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def append_tfrecord(writer, image, Hcw, params):
    if params["lens"] == 0:
        # This is just the horizontal field of view
        h_fov = params["FOV"]["x"]

        projection = "RECTILINEAR"
        fov = (
            h_fov
            * np.sqrt(
                params["imageSizePixels"]["x"] ** 2
                + params["imageSizePixels"]["y"] ** 2
            )
            / params["imageSizePixels"]["y"]
        )
        focal_length = params["imageSizePixels"]["y"] / (2 * np.tan(h_fov * 0.5))

    else:
        projection = "EQUISOLID"
        fov = params["FOV"]["x"]
        focal_length = 1.0 / params["radial"]["radiansPerPixel"]

    lens_centre = (params["centreOffset"]["x"], params["centreOffset"]["y"])
    # fmt: off
    transform = [
        Hcw.x.x, Hcw.x.y, Hcw.x.z, Hcw.x.t,
        Hcw.y.x, Hcw.y.y, Hcw.y.z, Hcw.y.t,
        Hcw.z.x, Hcw.z.y, Hcw.z.z, Hcw.z.t,
        Hcw.t.x, Hcw.t.y, Hcw.t.z, Hcw.t.t,
    ]
    # fmt: on

    features = {
        "image": bytes_feature(image),
        "lens/projection": bytes_feature(projection.encode("utf-8")),
        "lens/fov": float_feature(fov),
        "lens/focal_length": float_feature(focal_length),
        "lens/centre": float_list_feature(lens_centre),
        "mesh/Hcw": float_list_feature(transform),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(example.SerializeToString())


def decode(input_file, filter=None):

    # Now open the passed file
    with gzip.open(input_file, "rb") if input_file.endswith(
        "nbz"
    ) or input_file.endswith(".gz") else open(input_file, "rb") as f:
        # Open another file
        with open(input_file.replace("nbs", "json"), "w") as of:
            with tqdm(
                total=os.path.getsize(input_file), unit="b", unit_scale=True
            ) as progress:
                # While we can read a header
                while len(f.read(3)) == 3:
                    # Read our size
                    size = struct.unpack("<I", f.read(4))[0]

                    # Read our payload
                    payload = f.read(size)

                    # Update our progress bar
                    progress.update(7 + size)

                    # Read our timestamp
                    timestamp = struct.unpack("<Q", payload[:8])[0]

                    # Read our hash
                    type_hash = payload[8:16]

                    # If we know how to parse this type, parse it
                    if type_hash in parsers:
                        if filter is None or parsers[type_hash][0] in filter:
                            yield (
                                parsers[type_hash][0],
                                parsers[type_hash][1].FromString(payload[16:]),
                            )


if __name__ == "__main__":

    input_file = sys.argv[1]

    # First things first, check for a CameraParameters message
    # There is likely only one of these in the entire file and
    # it will apply to every image message in the file.
    params = None
    for msg in tqdm(decode(input_file, filter=["message.input.CameraParameters"])):
        # Found it!
        if params is None:
            params = MessageToDict(
                msg[1],
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                use_integers_for_enums=True,
            )
            print("\nFound a CameraParameters message.")
            print(params)
            break

    # No CameraParameters :(
    # Time to make things up :)
    if params is None:
        params = {
            "imageSizePixels": {"x": 1280, "y": 1024},
            "FOV": {"x": 3.14, "y": 3.14},
            "centreOffset": {"x": 0, "y": 0},
            "lens": 1,
            "radial": {"radiansPerPixel": 0.0026768},
        }
        print("\nFailed to find a CameraParameters message. Making one up.")
        print(params)

    # Determine output file names
    output_tfrecord = input_file.replace(".nbs", ".tfrecord")

    # Create the tfrecord writer
    writer = tf.python_io.TFRecordWriter(
        output_tfrecord,
        tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP),
    )

    for msg in tqdm(decode(input_file, filter=["message.input.Image"])):
        if msg[0] == "message.input.Image":
            # Convert image data into numpy array
            data = np.frombuffer(msg[1].data, np.uint8).reshape(
                (msg[1].dimensions.y, msg[1].dimensions.x)
            )

            # Convert the image into an RGB format
            if msg[1].format == 1195528775:  # GRBG
                image = cv2.cvtColor(data, cv2.COLOR_BayerGR2RGB)
            elif msg[1].format == 1111967570:  # RGGB
                image = cv2.cvtColor(data, cv2.COLOR_BayerRG2RGB)
            elif msg[1].format == 1196573255:  # GBRG
                image = cv2.cvtColor(data, cv2.COLOR_BayerGB2RGB)
            elif msg[1].format == 1380403010:  # BGGR
                image = cv2.cvtColor(data, cv2.COLOR_BayerBG2RGB)
            # Its not a known Bayer format, assume its already RGB
            # TODO: Handle other types here
            else:
                image = data

            # Encode the image as a png
            image = cv2.imencode(".png", image)[1].tostring()

            append_tfrecord(writer, image, msg[1].Hcw, params)

    writer.close()
