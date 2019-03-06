# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/motion/DiveCommand.proto

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


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/motion/DiveCommand.proto',
  package='protobuf.message.motion',
  syntax='proto3',
  serialized_pb=_b('\n message/motion/DiveCommand.proto\x12\x17protobuf.message.motion\x1a\x0cVector.proto\"\'\n\x0b\x44iveCommand\x12\x18\n\tdirection\x18\x01 \x01(\x0b\x32\x05.vec2\"\x0e\n\x0c\x44iveFinishedb\x06proto3')
  ,
  dependencies=[Vector__pb2.DESCRIPTOR,])




_DIVECOMMAND = _descriptor.Descriptor(
  name='DiveCommand',
  full_name='protobuf.message.motion.DiveCommand',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='direction', full_name='protobuf.message.motion.DiveCommand.direction', index=0,
      number=1, type=11, cpp_type=10, label=1,
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
  serialized_start=75,
  serialized_end=114,
)


_DIVEFINISHED = _descriptor.Descriptor(
  name='DiveFinished',
  full_name='protobuf.message.motion.DiveFinished',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
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
  serialized_start=116,
  serialized_end=130,
)

_DIVECOMMAND.fields_by_name['direction'].message_type = Vector__pb2._VEC2
DESCRIPTOR.message_types_by_name['DiveCommand'] = _DIVECOMMAND
DESCRIPTOR.message_types_by_name['DiveFinished'] = _DIVEFINISHED
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DiveCommand = _reflection.GeneratedProtocolMessageType('DiveCommand', (_message.Message,), dict(
  DESCRIPTOR = _DIVECOMMAND,
  __module__ = 'message.motion.DiveCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.DiveCommand)
  ))
_sym_db.RegisterMessage(DiveCommand)

DiveFinished = _reflection.GeneratedProtocolMessageType('DiveFinished', (_message.Message,), dict(
  DESCRIPTOR = _DIVEFINISHED,
  __module__ = 'message.motion.DiveCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.DiveFinished)
  ))
_sym_db.RegisterMessage(DiveFinished)


# @@protoc_insertion_point(module_scope)