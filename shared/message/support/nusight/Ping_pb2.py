# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/support/nusight/Ping.proto

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
  name='message/support/nusight/Ping.proto',
  package='protobuf.message.support.nusight',
  syntax='proto3',
  serialized_pb=_b('\n\"message/support/nusight/Ping.proto\x12 protobuf.message.support.nusight\"\x14\n\x04Ping\x12\x0c\n\x04time\x18\x01 \x01(\x04\x62\x06proto3')
)




_PING = _descriptor.Descriptor(
  name='Ping',
  full_name='protobuf.message.support.nusight.Ping',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='time', full_name='protobuf.message.support.nusight.Ping.time', index=0,
      number=1, type=4, cpp_type=4, label=1,
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
  serialized_start=72,
  serialized_end=92,
)

DESCRIPTOR.message_types_by_name['Ping'] = _PING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Ping = _reflection.GeneratedProtocolMessageType('Ping', (_message.Message,), dict(
  DESCRIPTOR = _PING,
  __module__ = 'message.support.nusight.Ping_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.support.nusight.Ping)
  ))
_sym_db.RegisterMessage(Ping)


# @@protoc_insertion_point(module_scope)
