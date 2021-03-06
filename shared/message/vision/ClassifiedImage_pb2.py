# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/vision/ClassifiedImage.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import Neutron_pb2 as Neutron__pb2
import Vector_pb2 as Vector__pb2
from message import Geometry_pb2 as message_dot_Geometry__pb2
from message.input import Sensors_pb2 as message_dot_input_dot_Sensors__pb2
from message.input import Image_pb2 as message_dot_input_dot_Image__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/vision/ClassifiedImage.proto',
  package='protobuf.message.vision',
  syntax='proto3',
  serialized_pb=_b('\n$message/vision/ClassifiedImage.proto\x12\x17protobuf.message.vision\x1a\rNeutron.proto\x1a\x0cVector.proto\x1a\x16message/Geometry.proto\x1a\x1bmessage/input/Sensors.proto\x1a\x19message/input/Image.proto\"\xfe\x06\n\x0f\x43lassifiedImage\x12\x36\n\x07sensors\x18\x01 \x01(\x0b\x32\x1f.protobuf.message.input.SensorsB\x04\x80\xb5\x18\x02\x12\x32\n\x05image\x18\x02 \x01(\x0b\x32\x1d.protobuf.message.input.ImageB\x04\x80\xb5\x18\x02\x12\x1a\n\ndimensions\x18\x03 \x01(\x0b\x32\x06.uvec2\x12Q\n\x0e\x62\x61llSeedPoints\x18\x04 \x03(\x0b\x32\x33.protobuf.message.vision.ClassifiedImage.SeedPointsB\x04\x88\xb5\x18\x03\x12\x1a\n\nballPoints\x18\x05 \x03(\x0b\x32\x06.ivec2\x12\x1d\n\x0ehorizon_normal\x18\x06 \x01(\x0b\x32\x05.vec3\x12\x1d\n\rvisualHorizon\x18\x07 \x03(\x0b\x32\x06.ivec2\x12L\n\x12horizontalSegments\x18\x08 \x03(\x0b\x32\x30.protobuf.message.vision.ClassifiedImage.Segment\x12J\n\x10verticalSegments\x18\t \x03(\x0b\x32\x30.protobuf.message.vision.ClassifiedImage.Segment\x12\'\n\x07horizon\x18\n \x01(\x0b\x32\x16.protobuf.message.Line\x1a\xdf\x01\n\x07Segment\x12K\n\x0csegmentClass\x18\x01 \x01(\x0e\x32\x35.protobuf.message.vision.ClassifiedImage.SegmentClass\x12\x0e\n\x06length\x18\x02 \x01(\r\x12\x11\n\tsubsample\x18\x03 \x01(\r\x12\x15\n\x05start\x18\x04 \x01(\x0b\x32\x06.ivec2\x12\x13\n\x03\x65nd\x18\x05 \x01(\x0b\x32\x06.ivec2\x12\x18\n\x08midpoint\x18\x06 \x01(\x0b\x32\x06.ivec2\x12\x10\n\x08previous\x18\x07 \x01(\x05\x12\x0c\n\x04next\x18\x08 \x01(\x05\x1a$\n\nSeedPoints\x12\x16\n\x06points\x18\x01 \x03(\x0b\x32\x06.ivec2\"k\n\x0cSegmentClass\x12\x11\n\rUNKNOWN_CLASS\x10\x00\x12\t\n\x05\x46IELD\x10\x01\x12\x08\n\x04\x42\x41LL\x10\x02\x12\x08\n\x04GOAL\x10\x03\x12\x08\n\x04LINE\x10\x04\x12\r\n\tCYAN_TEAM\x10\x05\x12\x10\n\x0cMAGENTA_TEAM\x10\x06\x62\x06proto3')
  ,
  dependencies=[Neutron__pb2.DESCRIPTOR,Vector__pb2.DESCRIPTOR,message_dot_Geometry__pb2.DESCRIPTOR,message_dot_input_dot_Sensors__pb2.DESCRIPTOR,message_dot_input_dot_Image__pb2.DESCRIPTOR,])



_CLASSIFIEDIMAGE_SEGMENTCLASS = _descriptor.EnumDescriptor(
  name='SegmentClass',
  full_name='protobuf.message.vision.ClassifiedImage.SegmentClass',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_CLASS', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIELD', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BALL', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GOAL', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LINE', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CYAN_TEAM', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MAGENTA_TEAM', index=6, number=6,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=962,
  serialized_end=1069,
)
_sym_db.RegisterEnumDescriptor(_CLASSIFIEDIMAGE_SEGMENTCLASS)


_CLASSIFIEDIMAGE_SEGMENT = _descriptor.Descriptor(
  name='Segment',
  full_name='protobuf.message.vision.ClassifiedImage.Segment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='segmentClass', full_name='protobuf.message.vision.ClassifiedImage.Segment.segmentClass', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='length', full_name='protobuf.message.vision.ClassifiedImage.Segment.length', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='subsample', full_name='protobuf.message.vision.ClassifiedImage.Segment.subsample', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start', full_name='protobuf.message.vision.ClassifiedImage.Segment.start', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end', full_name='protobuf.message.vision.ClassifiedImage.Segment.end', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='midpoint', full_name='protobuf.message.vision.ClassifiedImage.Segment.midpoint', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='previous', full_name='protobuf.message.vision.ClassifiedImage.Segment.previous', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='next', full_name='protobuf.message.vision.ClassifiedImage.Segment.next', index=7,
      number=8, type=5, cpp_type=1, label=1,
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
  serialized_start=699,
  serialized_end=922,
)

_CLASSIFIEDIMAGE_SEEDPOINTS = _descriptor.Descriptor(
  name='SeedPoints',
  full_name='protobuf.message.vision.ClassifiedImage.SeedPoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='points', full_name='protobuf.message.vision.ClassifiedImage.SeedPoints.points', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=924,
  serialized_end=960,
)

_CLASSIFIEDIMAGE = _descriptor.Descriptor(
  name='ClassifiedImage',
  full_name='protobuf.message.vision.ClassifiedImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sensors', full_name='protobuf.message.vision.ClassifiedImage.sensors', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\200\265\030\002')), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='protobuf.message.vision.ClassifiedImage.image', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\200\265\030\002')), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dimensions', full_name='protobuf.message.vision.ClassifiedImage.dimensions', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ballSeedPoints', full_name='protobuf.message.vision.ClassifiedImage.ballSeedPoints', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\210\265\030\003')), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ballPoints', full_name='protobuf.message.vision.ClassifiedImage.ballPoints', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='horizon_normal', full_name='protobuf.message.vision.ClassifiedImage.horizon_normal', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='visualHorizon', full_name='protobuf.message.vision.ClassifiedImage.visualHorizon', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='horizontalSegments', full_name='protobuf.message.vision.ClassifiedImage.horizontalSegments', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='verticalSegments', full_name='protobuf.message.vision.ClassifiedImage.verticalSegments', index=8,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='horizon', full_name='protobuf.message.vision.ClassifiedImage.horizon', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_CLASSIFIEDIMAGE_SEGMENT, _CLASSIFIEDIMAGE_SEEDPOINTS, ],
  enum_types=[
    _CLASSIFIEDIMAGE_SEGMENTCLASS,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=175,
  serialized_end=1069,
)

_CLASSIFIEDIMAGE_SEGMENT.fields_by_name['segmentClass'].enum_type = _CLASSIFIEDIMAGE_SEGMENTCLASS
_CLASSIFIEDIMAGE_SEGMENT.fields_by_name['start'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE_SEGMENT.fields_by_name['end'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE_SEGMENT.fields_by_name['midpoint'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE_SEGMENT.containing_type = _CLASSIFIEDIMAGE
_CLASSIFIEDIMAGE_SEEDPOINTS.fields_by_name['points'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE_SEEDPOINTS.containing_type = _CLASSIFIEDIMAGE
_CLASSIFIEDIMAGE.fields_by_name['sensors'].message_type = message_dot_input_dot_Sensors__pb2._SENSORS
_CLASSIFIEDIMAGE.fields_by_name['image'].message_type = message_dot_input_dot_Image__pb2._IMAGE
_CLASSIFIEDIMAGE.fields_by_name['dimensions'].message_type = Vector__pb2._UVEC2
_CLASSIFIEDIMAGE.fields_by_name['ballSeedPoints'].message_type = _CLASSIFIEDIMAGE_SEEDPOINTS
_CLASSIFIEDIMAGE.fields_by_name['ballPoints'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE.fields_by_name['horizon_normal'].message_type = Vector__pb2._VEC3
_CLASSIFIEDIMAGE.fields_by_name['visualHorizon'].message_type = Vector__pb2._IVEC2
_CLASSIFIEDIMAGE.fields_by_name['horizontalSegments'].message_type = _CLASSIFIEDIMAGE_SEGMENT
_CLASSIFIEDIMAGE.fields_by_name['verticalSegments'].message_type = _CLASSIFIEDIMAGE_SEGMENT
_CLASSIFIEDIMAGE.fields_by_name['horizon'].message_type = message_dot_Geometry__pb2._LINE
_CLASSIFIEDIMAGE_SEGMENTCLASS.containing_type = _CLASSIFIEDIMAGE
DESCRIPTOR.message_types_by_name['ClassifiedImage'] = _CLASSIFIEDIMAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ClassifiedImage = _reflection.GeneratedProtocolMessageType('ClassifiedImage', (_message.Message,), dict(

  Segment = _reflection.GeneratedProtocolMessageType('Segment', (_message.Message,), dict(
    DESCRIPTOR = _CLASSIFIEDIMAGE_SEGMENT,
    __module__ = 'message.vision.ClassifiedImage_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.vision.ClassifiedImage.Segment)
    ))
  ,

  SeedPoints = _reflection.GeneratedProtocolMessageType('SeedPoints', (_message.Message,), dict(
    DESCRIPTOR = _CLASSIFIEDIMAGE_SEEDPOINTS,
    __module__ = 'message.vision.ClassifiedImage_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.vision.ClassifiedImage.SeedPoints)
    ))
  ,
  DESCRIPTOR = _CLASSIFIEDIMAGE,
  __module__ = 'message.vision.ClassifiedImage_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.vision.ClassifiedImage)
  ))
_sym_db.RegisterMessage(ClassifiedImage)
_sym_db.RegisterMessage(ClassifiedImage.Segment)
_sym_db.RegisterMessage(ClassifiedImage.SeedPoints)


_CLASSIFIEDIMAGE.fields_by_name['sensors'].has_options = True
_CLASSIFIEDIMAGE.fields_by_name['sensors']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\200\265\030\002'))
_CLASSIFIEDIMAGE.fields_by_name['image'].has_options = True
_CLASSIFIEDIMAGE.fields_by_name['image']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\200\265\030\002'))
_CLASSIFIEDIMAGE.fields_by_name['ballSeedPoints'].has_options = True
_CLASSIFIEDIMAGE.fields_by_name['ballSeedPoints']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\210\265\030\003'))
# @@protoc_insertion_point(module_scope)
