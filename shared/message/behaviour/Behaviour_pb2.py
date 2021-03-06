# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/behaviour/Behaviour.proto

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
  name='message/behaviour/Behaviour.proto',
  package='protobuf.message.behaviour',
  syntax='proto3',
  serialized_pb=_b('\n!message/behaviour/Behaviour.proto\x12\x1aprotobuf.message.behaviour\"\xb4\x02\n\tBehaviour\x12:\n\x05state\x18\x01 \x01(\x0e\x32+.protobuf.message.behaviour.Behaviour.State\"\xea\x01\n\x05State\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x08\n\x04INIT\x10\x01\x12\x13\n\x0fSEARCH_FOR_BALL\x10\x02\x12\x14\n\x10SEARCH_FOR_GOALS\x10\x03\x12\x10\n\x0cWALK_TO_BALL\x10\x04\x12\r\n\tPICKED_UP\x10\x05\x12\x0b\n\x07INITIAL\x10\x06\x12\t\n\x05READY\x10\x07\x12\x07\n\x03SET\x10\x08\x12\x0b\n\x07TIMEOUT\x10\t\x12\x0c\n\x08\x46INISHED\x10\n\x12\r\n\tPENALISED\x10\x0b\x12\x0f\n\x0bGOALIE_WALK\x10\x0c\x12\x12\n\x0eMOVE_TO_CENTRE\x10\r\x12\x0e\n\nLOCALISING\x10\x0e\x62\x06proto3')
)



_BEHAVIOUR_STATE = _descriptor.EnumDescriptor(
  name='State',
  full_name='protobuf.message.behaviour.Behaviour.State',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INIT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SEARCH_FOR_BALL', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SEARCH_FOR_GOALS', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WALK_TO_BALL', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PICKED_UP', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INITIAL', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='READY', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SET', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TIMEOUT', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FINISHED', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PENALISED', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GOALIE_WALK', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MOVE_TO_CENTRE', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOCALISING', index=14, number=14,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=140,
  serialized_end=374,
)
_sym_db.RegisterEnumDescriptor(_BEHAVIOUR_STATE)


_BEHAVIOUR = _descriptor.Descriptor(
  name='Behaviour',
  full_name='protobuf.message.behaviour.Behaviour',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='protobuf.message.behaviour.Behaviour.state', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _BEHAVIOUR_STATE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=66,
  serialized_end=374,
)

_BEHAVIOUR.fields_by_name['state'].enum_type = _BEHAVIOUR_STATE
_BEHAVIOUR_STATE.containing_type = _BEHAVIOUR
DESCRIPTOR.message_types_by_name['Behaviour'] = _BEHAVIOUR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Behaviour = _reflection.GeneratedProtocolMessageType('Behaviour', (_message.Message,), dict(
  DESCRIPTOR = _BEHAVIOUR,
  __module__ = 'message.behaviour.Behaviour_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.behaviour.Behaviour)
  ))
_sym_db.RegisterMessage(Behaviour)


# @@protoc_insertion_point(module_scope)
