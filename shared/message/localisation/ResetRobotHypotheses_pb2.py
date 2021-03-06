# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/localisation/ResetRobotHypotheses.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Matrix_pb2 as Matrix__pb2
import Vector_pb2 as Vector__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/localisation/ResetRobotHypotheses.proto',
  package='protobuf.message.localisation',
  syntax='proto3',
  serialized_pb=_b('\n/message/localisation/ResetRobotHypotheses.proto\x12\x1dprotobuf.message.localisation\x1a\x0cMatrix.proto\x1a\x0cVector.proto\"\xdd\x01\n\x14ResetRobotHypotheses\x12L\n\nhypotheses\x18\x01 \x03(\x0b\x32\x38.protobuf.message.localisation.ResetRobotHypotheses.Self\x1aw\n\x04Self\x12\x17\n\x08position\x18\x01 \x01(\x0b\x32\x05.vec2\x12\x1b\n\x0cposition_cov\x18\x02 \x01(\x0b\x32\x05.mat2\x12\x0f\n\x07heading\x18\x03 \x01(\x01\x12\x13\n\x0bheading_var\x18\x04 \x01(\x01\x12\x13\n\x0b\x61\x62soluteYaw\x18\x05 \x01(\x08\x62\x06proto3')
  ,
  dependencies=[Matrix__pb2.DESCRIPTOR,Vector__pb2.DESCRIPTOR,])




_RESETROBOTHYPOTHESES_SELF = _descriptor.Descriptor(
  name='Self',
  full_name='protobuf.message.localisation.ResetRobotHypotheses.Self',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='protobuf.message.localisation.ResetRobotHypotheses.Self.position', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='position_cov', full_name='protobuf.message.localisation.ResetRobotHypotheses.Self.position_cov', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heading', full_name='protobuf.message.localisation.ResetRobotHypotheses.Self.heading', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='heading_var', full_name='protobuf.message.localisation.ResetRobotHypotheses.Self.heading_var', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='absoluteYaw', full_name='protobuf.message.localisation.ResetRobotHypotheses.Self.absoluteYaw', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=213,
  serialized_end=332,
)

_RESETROBOTHYPOTHESES = _descriptor.Descriptor(
  name='ResetRobotHypotheses',
  full_name='protobuf.message.localisation.ResetRobotHypotheses',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='hypotheses', full_name='protobuf.message.localisation.ResetRobotHypotheses.hypotheses', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RESETROBOTHYPOTHESES_SELF, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=111,
  serialized_end=332,
)

_RESETROBOTHYPOTHESES_SELF.fields_by_name['position'].message_type = Vector__pb2._VEC2
_RESETROBOTHYPOTHESES_SELF.fields_by_name['position_cov'].message_type = Matrix__pb2._MAT2
_RESETROBOTHYPOTHESES_SELF.containing_type = _RESETROBOTHYPOTHESES
_RESETROBOTHYPOTHESES.fields_by_name['hypotheses'].message_type = _RESETROBOTHYPOTHESES_SELF
DESCRIPTOR.message_types_by_name['ResetRobotHypotheses'] = _RESETROBOTHYPOTHESES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ResetRobotHypotheses = _reflection.GeneratedProtocolMessageType('ResetRobotHypotheses', (_message.Message,), dict(

  Self = _reflection.GeneratedProtocolMessageType('Self', (_message.Message,), dict(
    DESCRIPTOR = _RESETROBOTHYPOTHESES_SELF,
    __module__ = 'message.localisation.ResetRobotHypotheses_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.localisation.ResetRobotHypotheses.Self)
    ))
  ,
  DESCRIPTOR = _RESETROBOTHYPOTHESES,
  __module__ = 'message.localisation.ResetRobotHypotheses_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.localisation.ResetRobotHypotheses)
  ))
_sym_db.RegisterMessage(ResetRobotHypotheses)
_sym_db.RegisterMessage(ResetRobotHypotheses.Self)


# @@protoc_insertion_point(module_scope)
