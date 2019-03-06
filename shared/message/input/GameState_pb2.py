# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/input/GameState.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/input/GameState.proto',
  package='protobuf.message.input',
  syntax='proto3',
  serialized_pb=_b('\n\x1dmessage/input/GameState.proto\x12\x16protobuf.message.input\x1a\x1fgoogle/protobuf/timestamp.proto\"\xc1\n\n\tGameState\x12\x34\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32&.protobuf.message.input.GameState.Data\x12\r\n\x05\x65vent\x18\x02 \x01(\t\x1a\xee\t\n\x04\x44\x61ta\x12;\n\x05phase\x18\x01 \x01(\x0e\x32,.protobuf.message.input.GameState.Data.Phase\x12\x39\n\x04mode\x18\x02 \x01(\x0e\x32+.protobuf.message.input.GameState.Data.Mode\x12\x12\n\nfirst_half\x18\x03 \x01(\x08\x12\x18\n\x10kicked_out_by_us\x18\x04 \x01(\x08\x12\x33\n\x0fkicked_out_time\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x14\n\x0cour_kick_off\x18\x06 \x01(\x08\x12\x30\n\x0cprimary_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\x0esecondary_time\x18\x08 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x39\n\x04team\x18\t \x01(\x0b\x32+.protobuf.message.input.GameState.Data.Team\x12=\n\x08opponent\x18\n \x01(\x0b\x32+.protobuf.message.input.GameState.Data.Team\x12:\n\x04self\x18\x0b \x01(\x0b\x32,.protobuf.message.input.GameState.Data.Robot\x1a\x92\x01\n\x05Robot\x12\n\n\x02id\x18\x01 \x01(\r\x12L\n\x0epenalty_reason\x18\x02 \x01(\x0e\x32\x34.protobuf.message.input.GameState.Data.PenaltyReason\x12/\n\x0bunpenalised\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1a|\n\x04Team\x12\x0f\n\x07team_id\x18\x01 \x01(\r\x12\r\n\x05score\x18\x02 \x01(\r\x12\x15\n\rcoach_message\x18\x03 \x01(\t\x12=\n\x07players\x18\x04 \x03(\x0b\x32,.protobuf.message.input.GameState.Data.Robot\"H\n\x04Mode\x12\x10\n\x0cUNKNOWN_MODE\x10\x00\x12\n\n\x06NORMAL\x10\x01\x12\x14\n\x10PENALTY_SHOOTOUT\x10\x02\x12\x0c\n\x08OVERTIME\x10\x03\"c\n\x05Phase\x12\x11\n\rUNKNOWN_PHASE\x10\x00\x12\x0b\n\x07INITIAL\x10\x01\x12\t\n\x05READY\x10\x02\x12\x07\n\x03SET\x10\x03\x12\x0b\n\x07PLAYING\x10\x04\x12\x0b\n\x07TIMEOUT\x10\x05\x12\x0c\n\x08\x46INISHED\x10\x06\"\x96\x02\n\rPenaltyReason\x12\x1a\n\x16UNKNOWN_PENALTY_REASON\x10\x00\x12\x0f\n\x0bUNPENALISED\x10\x01\x12\x15\n\x11\x42\x41LL_MANIPULATION\x10\x02\x12\x14\n\x10PHYSICAL_CONTACT\x10\x03\x12\x12\n\x0eILLEGAL_ATTACK\x10\x04\x12\x13\n\x0fILLEGAL_DEFENSE\x10\x05\x12\x16\n\x12REQUEST_FOR_PICKUP\x10\x06\x12\x17\n\x13REQUEST_FOR_SERVICE\x10\x07\x12!\n\x1dREQUEST_FOR_PICKUP_TO_SERVICE\x10\x08\x12\x0e\n\nSUBSTITUTE\x10\t\x12\n\n\x06MANUAL\x10\n\x12\x12\n\x0ePLAYER_PUSHING\x10\x0b\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])



_GAMESTATE_DATA_MODE = _descriptor.EnumDescriptor(
  name='Mode',
  full_name='protobuf.message.input.GameState.Data.Mode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_MODE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NORMAL', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PENALTY_SHOOTOUT', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OVERTIME', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=982,
  serialized_end=1054,
)
_sym_db.RegisterEnumDescriptor(_GAMESTATE_DATA_MODE)

_GAMESTATE_DATA_PHASE = _descriptor.EnumDescriptor(
  name='Phase',
  full_name='protobuf.message.input.GameState.Data.Phase',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_PHASE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INITIAL', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='READY', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SET', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PLAYING', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TIMEOUT', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FINISHED', index=6, number=6,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1056,
  serialized_end=1155,
)
_sym_db.RegisterEnumDescriptor(_GAMESTATE_DATA_PHASE)

_GAMESTATE_DATA_PENALTYREASON = _descriptor.EnumDescriptor(
  name='PenaltyReason',
  full_name='protobuf.message.input.GameState.Data.PenaltyReason',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_PENALTY_REASON', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNPENALISED', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BALL_MANIPULATION', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PHYSICAL_CONTACT', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ILLEGAL_ATTACK', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ILLEGAL_DEFENSE', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_FOR_PICKUP', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_FOR_SERVICE', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REQUEST_FOR_PICKUP_TO_SERVICE', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUBSTITUTE', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MANUAL', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PLAYER_PUSHING', index=11, number=11,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1158,
  serialized_end=1436,
)
_sym_db.RegisterEnumDescriptor(_GAMESTATE_DATA_PENALTYREASON)


_GAMESTATE_DATA_ROBOT = _descriptor.Descriptor(
  name='Robot',
  full_name='protobuf.message.input.GameState.Data.Robot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='protobuf.message.input.GameState.Data.Robot.id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='penalty_reason', full_name='protobuf.message.input.GameState.Data.Robot.penalty_reason', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='unpenalised', full_name='protobuf.message.input.GameState.Data.Robot.unpenalised', index=2,
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
  serialized_start=708,
  serialized_end=854,
)

_GAMESTATE_DATA_TEAM = _descriptor.Descriptor(
  name='Team',
  full_name='protobuf.message.input.GameState.Data.Team',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='team_id', full_name='protobuf.message.input.GameState.Data.Team.team_id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score', full_name='protobuf.message.input.GameState.Data.Team.score', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='coach_message', full_name='protobuf.message.input.GameState.Data.Team.coach_message', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='players', full_name='protobuf.message.input.GameState.Data.Team.players', index=3,
      number=4, type=11, cpp_type=10, label=3,
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
  serialized_start=856,
  serialized_end=980,
)

_GAMESTATE_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='protobuf.message.input.GameState.Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phase', full_name='protobuf.message.input.GameState.Data.phase', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mode', full_name='protobuf.message.input.GameState.Data.mode', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_half', full_name='protobuf.message.input.GameState.Data.first_half', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kicked_out_by_us', full_name='protobuf.message.input.GameState.Data.kicked_out_by_us', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kicked_out_time', full_name='protobuf.message.input.GameState.Data.kicked_out_time', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='our_kick_off', full_name='protobuf.message.input.GameState.Data.our_kick_off', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='primary_time', full_name='protobuf.message.input.GameState.Data.primary_time', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='secondary_time', full_name='protobuf.message.input.GameState.Data.secondary_time', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='team', full_name='protobuf.message.input.GameState.Data.team', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='opponent', full_name='protobuf.message.input.GameState.Data.opponent', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='self', full_name='protobuf.message.input.GameState.Data.self', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GAMESTATE_DATA_ROBOT, _GAMESTATE_DATA_TEAM, ],
  enum_types=[
    _GAMESTATE_DATA_MODE,
    _GAMESTATE_DATA_PHASE,
    _GAMESTATE_DATA_PENALTYREASON,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=174,
  serialized_end=1436,
)

_GAMESTATE = _descriptor.Descriptor(
  name='GameState',
  full_name='protobuf.message.input.GameState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='protobuf.message.input.GameState.data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='event', full_name='protobuf.message.input.GameState.event', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GAMESTATE_DATA, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=1436,
)

_GAMESTATE_DATA_ROBOT.fields_by_name['penalty_reason'].enum_type = _GAMESTATE_DATA_PENALTYREASON
_GAMESTATE_DATA_ROBOT.fields_by_name['unpenalised'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMESTATE_DATA_ROBOT.containing_type = _GAMESTATE_DATA
_GAMESTATE_DATA_TEAM.fields_by_name['players'].message_type = _GAMESTATE_DATA_ROBOT
_GAMESTATE_DATA_TEAM.containing_type = _GAMESTATE_DATA
_GAMESTATE_DATA.fields_by_name['phase'].enum_type = _GAMESTATE_DATA_PHASE
_GAMESTATE_DATA.fields_by_name['mode'].enum_type = _GAMESTATE_DATA_MODE
_GAMESTATE_DATA.fields_by_name['kicked_out_time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMESTATE_DATA.fields_by_name['primary_time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMESTATE_DATA.fields_by_name['secondary_time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMESTATE_DATA.fields_by_name['team'].message_type = _GAMESTATE_DATA_TEAM
_GAMESTATE_DATA.fields_by_name['opponent'].message_type = _GAMESTATE_DATA_TEAM
_GAMESTATE_DATA.fields_by_name['self'].message_type = _GAMESTATE_DATA_ROBOT
_GAMESTATE_DATA.containing_type = _GAMESTATE
_GAMESTATE_DATA_MODE.containing_type = _GAMESTATE_DATA
_GAMESTATE_DATA_PHASE.containing_type = _GAMESTATE_DATA
_GAMESTATE_DATA_PENALTYREASON.containing_type = _GAMESTATE_DATA
_GAMESTATE.fields_by_name['data'].message_type = _GAMESTATE_DATA
DESCRIPTOR.message_types_by_name['GameState'] = _GAMESTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GameState = _reflection.GeneratedProtocolMessageType('GameState', (_message.Message,), dict(

  Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), dict(

    Robot = _reflection.GeneratedProtocolMessageType('Robot', (_message.Message,), dict(
      DESCRIPTOR = _GAMESTATE_DATA_ROBOT,
      __module__ = 'message.input.GameState_pb2'
      # @@protoc_insertion_point(class_scope:protobuf.message.input.GameState.Data.Robot)
      ))
    ,

    Team = _reflection.GeneratedProtocolMessageType('Team', (_message.Message,), dict(
      DESCRIPTOR = _GAMESTATE_DATA_TEAM,
      __module__ = 'message.input.GameState_pb2'
      # @@protoc_insertion_point(class_scope:protobuf.message.input.GameState.Data.Team)
      ))
    ,
    DESCRIPTOR = _GAMESTATE_DATA,
    __module__ = 'message.input.GameState_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameState.Data)
    ))
  ,
  DESCRIPTOR = _GAMESTATE,
  __module__ = 'message.input.GameState_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.input.GameState)
  ))
_sym_db.RegisterMessage(GameState)
_sym_db.RegisterMessage(GameState.Data)
_sym_db.RegisterMessage(GameState.Data.Robot)
_sym_db.RegisterMessage(GameState.Data.Team)


# @@protoc_insertion_point(module_scope)
