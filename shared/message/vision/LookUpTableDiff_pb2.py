# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/vision/LookUpTableDiff.proto

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
  name='message/vision/LookUpTableDiff.proto',
  package='protobuf.message.vision',
  syntax='proto3',
  serialized_pb=_b('\n$message/vision/LookUpTableDiff.proto\x12\x17protobuf.message.vision\"\x81\x01\n\x0fLookUpTableDiff\x12;\n\x04\x64iff\x18\x01 \x03(\x0b\x32-.protobuf.message.vision.LookUpTableDiff.Diff\x1a\x31\n\x04\x44iff\x12\x11\n\tlut_index\x18\x01 \x01(\r\x12\x16\n\x0e\x63lassification\x18\x02 \x01(\rb\x06proto3')
)




_LOOKUPTABLEDIFF_DIFF = _descriptor.Descriptor(
  name='Diff',
  full_name='protobuf.message.vision.LookUpTableDiff.Diff',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lut_index', full_name='protobuf.message.vision.LookUpTableDiff.Diff.lut_index', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classification', full_name='protobuf.message.vision.LookUpTableDiff.Diff.classification', index=1,
      number=2, type=13, cpp_type=3, label=1,
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
  serialized_start=146,
  serialized_end=195,
)

_LOOKUPTABLEDIFF = _descriptor.Descriptor(
  name='LookUpTableDiff',
  full_name='protobuf.message.vision.LookUpTableDiff',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='diff', full_name='protobuf.message.vision.LookUpTableDiff.diff', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_LOOKUPTABLEDIFF_DIFF, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=66,
  serialized_end=195,
)

_LOOKUPTABLEDIFF_DIFF.containing_type = _LOOKUPTABLEDIFF
_LOOKUPTABLEDIFF.fields_by_name['diff'].message_type = _LOOKUPTABLEDIFF_DIFF
DESCRIPTOR.message_types_by_name['LookUpTableDiff'] = _LOOKUPTABLEDIFF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LookUpTableDiff = _reflection.GeneratedProtocolMessageType('LookUpTableDiff', (_message.Message,), dict(

  Diff = _reflection.GeneratedProtocolMessageType('Diff', (_message.Message,), dict(
    DESCRIPTOR = _LOOKUPTABLEDIFF_DIFF,
    __module__ = 'message.vision.LookUpTableDiff_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.vision.LookUpTableDiff.Diff)
    ))
  ,
  DESCRIPTOR = _LOOKUPTABLEDIFF,
  __module__ = 'message.vision.LookUpTableDiff_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.vision.LookUpTableDiff)
  ))
_sym_db.RegisterMessage(LookUpTableDiff)
_sym_db.RegisterMessage(LookUpTableDiff.Diff)


# @@protoc_insertion_point(module_scope)