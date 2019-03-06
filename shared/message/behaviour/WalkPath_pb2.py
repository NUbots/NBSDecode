# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/behaviour/WalkPath.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from message.behaviour import MotionCommand_pb2 as message_dot_behaviour_dot_MotionCommand__pb2
import Vector_pb2 as Vector__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/behaviour/WalkPath.proto',
  package='protobuf.message.behaviour',
  syntax='proto3',
  serialized_pb=_b('\n message/behaviour/WalkPath.proto\x12\x1aprotobuf.message.behaviour\x1a%message/behaviour/MotionCommand.proto\x1a\x0cVector.proto\"\xa2\x01\n\x08WalkPath\x12\x15\n\x06states\x18\x01 \x03(\x0b\x32\x05.vec3\x12\x18\n\tballSpace\x18\x02 \x01(\x0b\x32\x05.vec3\x12\x14\n\x05start\x18\x03 \x01(\x0b\x32\x05.vec3\x12\x13\n\x04goal\x18\x04 \x01(\x0b\x32\x05.vec3\x12:\n\x07\x63ommand\x18\x05 \x01(\x0b\x32).protobuf.message.behaviour.MotionCommandb\x06proto3')
  ,
  dependencies=[message_dot_behaviour_dot_MotionCommand__pb2.DESCRIPTOR,Vector__pb2.DESCRIPTOR,])




_WALKPATH = _descriptor.Descriptor(
  name='WalkPath',
  full_name='protobuf.message.behaviour.WalkPath',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='states', full_name='protobuf.message.behaviour.WalkPath.states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ballSpace', full_name='protobuf.message.behaviour.WalkPath.ballSpace', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='protobuf.message.behaviour.WalkPath.start', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goal', full_name='protobuf.message.behaviour.WalkPath.goal', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='command', full_name='protobuf.message.behaviour.WalkPath.command', index=4,
      number=5, type=11, cpp_type=10, label=1,
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
  serialized_start=118,
  serialized_end=280,
)

_WALKPATH.fields_by_name['states'].message_type = Vector__pb2._VEC3
_WALKPATH.fields_by_name['ballSpace'].message_type = Vector__pb2._VEC3
_WALKPATH.fields_by_name['start'].message_type = Vector__pb2._VEC3
_WALKPATH.fields_by_name['goal'].message_type = Vector__pb2._VEC3
_WALKPATH.fields_by_name['command'].message_type = message_dot_behaviour_dot_MotionCommand__pb2._MOTIONCOMMAND
DESCRIPTOR.message_types_by_name['WalkPath'] = _WALKPATH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WalkPath = _reflection.GeneratedProtocolMessageType('WalkPath', (_message.Message,), dict(
  DESCRIPTOR = _WALKPATH,
  __module__ = 'message.behaviour.WalkPath_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.behaviour.WalkPath)
  ))
_sym_db.RegisterMessage(WalkPath)


# @@protoc_insertion_point(module_scope)
