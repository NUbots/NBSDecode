# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/localisation/Ball.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Vector_pb2 as Vector__pb2
import Matrix_pb2 as Matrix__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/localisation/Ball.proto',
  package='protobuf.message.localisation',
  syntax='proto3',
  serialized_pb=_b('\n\x1fmessage/localisation/Ball.proto\x12\x1dprotobuf.message.localisation\x1a\x0cVector.proto\x1a\x0cMatrix.proto\":\n\x04\x42\x61ll\x12\x17\n\x08position\x18\x01 \x01(\x0b\x32\x05.vec2\x12\x19\n\ncovariance\x18\x02 \x01(\x0b\x32\x05.mat2b\x06proto3')
  ,
  dependencies=[Vector__pb2.DESCRIPTOR,Matrix__pb2.DESCRIPTOR,])




_BALL = _descriptor.Descriptor(
  name='Ball',
  full_name='protobuf.message.localisation.Ball',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='protobuf.message.localisation.Ball.position', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='covariance', full_name='protobuf.message.localisation.Ball.covariance', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=94,
  serialized_end=152,
)

_BALL.fields_by_name['position'].message_type = Vector__pb2._VEC2
_BALL.fields_by_name['covariance'].message_type = Matrix__pb2._MAT2
DESCRIPTOR.message_types_by_name['Ball'] = _BALL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Ball = _reflection.GeneratedProtocolMessageType('Ball', (_message.Message,), dict(
  DESCRIPTOR = _BALL,
  __module__ = 'message.localisation.Ball_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.localisation.Ball)
  ))
_sym_db.RegisterMessage(Ball)


# @@protoc_insertion_point(module_scope)