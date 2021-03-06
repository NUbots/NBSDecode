# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/support/nuclear/ReactionStatistics.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/support/nuclear/ReactionStatistics.proto',
  package='protobuf.message.support.nuclear',
  syntax='proto3',
  serialized_pb=_b('\n0message/support/nuclear/ReactionStatistics.proto\x12 protobuf.message.support.nuclear\"\xd3\x01\n\x12ReactionStatistics\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0btriggerName\x18\x02 \x01(\t\x12\x14\n\x0c\x66unctionName\x18\x03 \x01(\t\x12\x12\n\nreactionId\x18\x04 \x01(\x04\x12\x0e\n\x06taskId\x18\x05 \x01(\x04\x12\x17\n\x0f\x63\x61useReactionId\x18\x06 \x01(\x04\x12\x13\n\x0b\x63\x61useTaskId\x18\x07 \x01(\x04\x12\x0f\n\x07\x65mitted\x18\x08 \x01(\x04\x12\x0f\n\x07started\x18\t \x01(\x04\x12\x10\n\x08\x66inished\x18\n \x01(\x04\x62\x06proto3')
)




_REACTIONSTATISTICS = _descriptor.Descriptor(
  name='ReactionStatistics',
  full_name='protobuf.message.support.nuclear.ReactionStatistics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='protobuf.message.support.nuclear.ReactionStatistics.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='triggerName', full_name='protobuf.message.support.nuclear.ReactionStatistics.triggerName', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='functionName', full_name='protobuf.message.support.nuclear.ReactionStatistics.functionName', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reactionId', full_name='protobuf.message.support.nuclear.ReactionStatistics.reactionId', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='taskId', full_name='protobuf.message.support.nuclear.ReactionStatistics.taskId', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='causeReactionId', full_name='protobuf.message.support.nuclear.ReactionStatistics.causeReactionId', index=5,
      number=6, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='causeTaskId', full_name='protobuf.message.support.nuclear.ReactionStatistics.causeTaskId', index=6,
      number=7, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='emitted', full_name='protobuf.message.support.nuclear.ReactionStatistics.emitted', index=7,
      number=8, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='started', full_name='protobuf.message.support.nuclear.ReactionStatistics.started', index=8,
      number=9, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='finished', full_name='protobuf.message.support.nuclear.ReactionStatistics.finished', index=9,
      number=10, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=87,
  serialized_end=298,
)

DESCRIPTOR.message_types_by_name['ReactionStatistics'] = _REACTIONSTATISTICS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReactionStatistics = _reflection.GeneratedProtocolMessageType('ReactionStatistics', (_message.Message,), dict(
  DESCRIPTOR = _REACTIONSTATISTICS,
  __module__ = 'message.support.nuclear.ReactionStatistics_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.support.nuclear.ReactionStatistics)
  ))
_sym_db.RegisterMessage(ReactionStatistics)


# @@protoc_insertion_point(module_scope)
