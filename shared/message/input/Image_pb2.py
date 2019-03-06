# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/input/Image.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
import Matrix_pb2 as Matrix__pb2
import Vector_pb2 as Vector__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/input/Image.proto',
  package='protobuf.message.input',
  syntax='proto3',
  serialized_pb=_b('\n\x19message/input/Image.proto\x12\x16protobuf.message.input\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x0cMatrix.proto\x1a\x0cVector.proto\"\x98\x02\n\x05Image\x12\x0e\n\x06\x66ormat\x18\x01 \x01(\r\x12\x1a\n\ndimensions\x18\x02 \x01(\x0b\x32\x06.uvec2\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x11\n\tcamera_id\x18\x04 \x01(\r\x12\x15\n\rserial_number\x18\x05 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x12\n\x03Hcw\x18\x07 \x01(\x0b\x32\x05.mat4\x12\x30\n\x04lens\x18\x08 \x01(\x0b\x32\".protobuf.message.input.Image.Lens\x1a\x36\n\x04Lens\".\n\nProjection\x12\x0f\n\x0bRECTILINEAR\x10\x00\x12\x0f\n\x0b\x45QUIDISTANT\x10\x01\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,Matrix__pb2.DESCRIPTOR,Vector__pb2.DESCRIPTOR,])



_IMAGE_LENS_PROJECTION = _descriptor.EnumDescriptor(
  name='Projection',
  full_name='protobuf.message.input.Image.Lens.Projection',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RECTILINEAR', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EQUIDISTANT', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=349,
  serialized_end=395,
)
_sym_db.RegisterEnumDescriptor(_IMAGE_LENS_PROJECTION)


_IMAGE_LENS = _descriptor.Descriptor(
  name='Lens',
  full_name='protobuf.message.input.Image.Lens',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _IMAGE_LENS_PROJECTION,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=341,
  serialized_end=395,
)

_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='protobuf.message.input.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='format', full_name='protobuf.message.input.Image.format', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dimensions', full_name='protobuf.message.input.Image.dimensions', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='protobuf.message.input.Image.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='camera_id', full_name='protobuf.message.input.Image.camera_id', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='serial_number', full_name='protobuf.message.input.Image.serial_number', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='protobuf.message.input.Image.timestamp', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Hcw', full_name='protobuf.message.input.Image.Hcw', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lens', full_name='protobuf.message.input.Image.lens', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_IMAGE_LENS, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=115,
  serialized_end=395,
)

_IMAGE_LENS.containing_type = _IMAGE
_IMAGE_LENS_PROJECTION.containing_type = _IMAGE_LENS
_IMAGE.fields_by_name['dimensions'].message_type = Vector__pb2._UVEC2
_IMAGE.fields_by_name['timestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_IMAGE.fields_by_name['Hcw'].message_type = Matrix__pb2._MAT4
_IMAGE.fields_by_name['lens'].message_type = _IMAGE_LENS
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(

  Lens = _reflection.GeneratedProtocolMessageType('Lens', (_message.Message,), dict(
    DESCRIPTOR = _IMAGE_LENS,
    __module__ = 'message.input.Image_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.Image.Lens)
    ))
  ,
  DESCRIPTOR = _IMAGE,
  __module__ = 'message.input.Image_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.input.Image)
  ))
_sym_db.RegisterMessage(Image)
_sym_db.RegisterMessage(Image.Lens)


# @@protoc_insertion_point(module_scope)