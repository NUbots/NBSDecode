# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/motion/TorsoMotionCommand.proto

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
  name='message/motion/TorsoMotionCommand.proto',
  package='protobuf.message.motion',
  syntax='proto3',
  serialized_pb=_b('\n\'message/motion/TorsoMotionCommand.proto\x12\x17protobuf.message.motion\x1a\x0cMatrix.proto\x1a\x0cVector.proto\"_\n\x11TorsoMotionUpdate\x12\x18\n\tframeArms\x18\x01 \x01(\x0b\x32\x05.vec3\x12\x18\n\tframeLegs\x18\x02 \x01(\x0b\x32\x05.vec3\x12\x16\n\x07\x66rame3D\x18\x03 \x01(\x0b\x32\x05.mat4\"J\n\x13TorsoPositionUpdate\x12\x17\n\x08position\x18\x01 \x01(\x0b\x32\x05.vec3\x12\x1a\n\x0b\x64\x65stination\x18\x02 \x01(\x0b\x32\x05.vec3\"\x13\n\x11\x45nableTorsoMotion\"\x14\n\x12\x44isableTorsoMotionb\x06proto3')
  ,
  dependencies=[Matrix__pb2.DESCRIPTOR,Vector__pb2.DESCRIPTOR,])




_TORSOMOTIONUPDATE = _descriptor.Descriptor(
  name='TorsoMotionUpdate',
  full_name='protobuf.message.motion.TorsoMotionUpdate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frameArms', full_name='protobuf.message.motion.TorsoMotionUpdate.frameArms', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frameLegs', full_name='protobuf.message.motion.TorsoMotionUpdate.frameLegs', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame3D', full_name='protobuf.message.motion.TorsoMotionUpdate.frame3D', index=2,
      number=3, type=11, cpp_type=10, label=1,
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
  serialized_start=96,
  serialized_end=191,
)


_TORSOPOSITIONUPDATE = _descriptor.Descriptor(
  name='TorsoPositionUpdate',
  full_name='protobuf.message.motion.TorsoPositionUpdate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='protobuf.message.motion.TorsoPositionUpdate.position', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='destination', full_name='protobuf.message.motion.TorsoPositionUpdate.destination', index=1,
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
  serialized_start=193,
  serialized_end=267,
)


_ENABLETORSOMOTION = _descriptor.Descriptor(
  name='EnableTorsoMotion',
  full_name='protobuf.message.motion.EnableTorsoMotion',
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
  serialized_start=269,
  serialized_end=288,
)


_DISABLETORSOMOTION = _descriptor.Descriptor(
  name='DisableTorsoMotion',
  full_name='protobuf.message.motion.DisableTorsoMotion',
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
  serialized_start=290,
  serialized_end=310,
)

_TORSOMOTIONUPDATE.fields_by_name['frameArms'].message_type = Vector__pb2._VEC3
_TORSOMOTIONUPDATE.fields_by_name['frameLegs'].message_type = Vector__pb2._VEC3
_TORSOMOTIONUPDATE.fields_by_name['frame3D'].message_type = Matrix__pb2._MAT4
_TORSOPOSITIONUPDATE.fields_by_name['position'].message_type = Vector__pb2._VEC3
_TORSOPOSITIONUPDATE.fields_by_name['destination'].message_type = Vector__pb2._VEC3
DESCRIPTOR.message_types_by_name['TorsoMotionUpdate'] = _TORSOMOTIONUPDATE
DESCRIPTOR.message_types_by_name['TorsoPositionUpdate'] = _TORSOPOSITIONUPDATE
DESCRIPTOR.message_types_by_name['EnableTorsoMotion'] = _ENABLETORSOMOTION
DESCRIPTOR.message_types_by_name['DisableTorsoMotion'] = _DISABLETORSOMOTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TorsoMotionUpdate = _reflection.GeneratedProtocolMessageType('TorsoMotionUpdate', (_message.Message,), dict(
  DESCRIPTOR = _TORSOMOTIONUPDATE,
  __module__ = 'message.motion.TorsoMotionCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.TorsoMotionUpdate)
  ))
_sym_db.RegisterMessage(TorsoMotionUpdate)

TorsoPositionUpdate = _reflection.GeneratedProtocolMessageType('TorsoPositionUpdate', (_message.Message,), dict(
  DESCRIPTOR = _TORSOPOSITIONUPDATE,
  __module__ = 'message.motion.TorsoMotionCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.TorsoPositionUpdate)
  ))
_sym_db.RegisterMessage(TorsoPositionUpdate)

EnableTorsoMotion = _reflection.GeneratedProtocolMessageType('EnableTorsoMotion', (_message.Message,), dict(
  DESCRIPTOR = _ENABLETORSOMOTION,
  __module__ = 'message.motion.TorsoMotionCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.EnableTorsoMotion)
  ))
_sym_db.RegisterMessage(EnableTorsoMotion)

DisableTorsoMotion = _reflection.GeneratedProtocolMessageType('DisableTorsoMotion', (_message.Message,), dict(
  DESCRIPTOR = _DISABLETORSOMOTION,
  __module__ = 'message.motion.TorsoMotionCommand_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.motion.DisableTorsoMotion)
  ))
_sym_db.RegisterMessage(DisableTorsoMotion)


# @@protoc_insertion_point(module_scope)
