# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message/input/GameEvents.proto

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
from message.input import GameState_pb2 as message_dot_input_dot_GameState__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='message/input/GameEvents.proto',
  package='protobuf.message.input',
  syntax='proto3',
  serialized_pb=_b('\n\x1emessage/input/GameEvents.proto\x12\x16protobuf.message.input\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1dmessage/input/GameState.proto\"\x8e\n\n\nGameEvents\x1a\x30\n\x05Score\x12\x10\n\x08ownScore\x18\x01 \x01(\r\x12\x15\n\ropponentScore\x18\x02 \x01(\r\x1a]\n\nGoalScored\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x12\x12\n\ntotalScore\x18\x02 \x01(\r\x1a\xcc\x01\n\x0cPenalisation\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x12\x0f\n\x07robotId\x18\x02 \x01(\r\x12(\n\x04\x65nds\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x44\n\x06reason\x18\x04 \x01(\x0e\x32\x34.protobuf.message.input.GameState.Data.PenaltyReason\x1a^\n\x0eUnpenalisation\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x12\x0f\n\x07robotId\x18\x02 \x01(\r\x1a\\\n\x0c\x43oachMessage\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x12\x0f\n\x07message\x18\x02 \x01(\t\x1a\x1d\n\x08HalfTime\x12\x11\n\tfirstHalf\x18\x01 \x01(\x08\x1av\n\rBallKickedOut\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x12(\n\x04time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1aJ\n\x0bKickOffTeam\x12;\n\x07\x63ontext\x18\x01 \x01(\x0e\x32*.protobuf.message.input.GameEvents.Context\x1a\xaa\x02\n\tGamePhase\x12;\n\x05phase\x18\x01 \x01(\x0e\x32,.protobuf.message.input.GameState.Data.Phase\x12-\n\treadyTime\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12+\n\x07\x65ndHalf\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x62\x61llFree\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12(\n\x04\x65nds\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08nextHalf\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x1a\x45\n\x08GameMode\x12\x39\n\x04mode\x18\x01 \x01(\x0e\x32+.protobuf.message.input.GameState.Data.Mode\"M\n\x07\x43ontext\x12\x13\n\x0fUNKNOWN_CONTEXT\x10\x00\x12\x08\n\x04SELF\x10\x01\x12\x08\n\x04TEAM\x10\x02\x12\x0c\n\x08OPPONENT\x10\x03\x12\x0b\n\x07UNKNOWN\x10\x04\"<\n\nTeamColour\x12\x17\n\x13UNKNOWN_TEAM_COLOUR\x10\x00\x12\x08\n\x04\x43YAN\x10\x01\x12\x0b\n\x07MAGENTA\x10\x02\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,message_dot_input_dot_GameState__pb2.DESCRIPTOR,])



_GAMEEVENTS_CONTEXT = _descriptor.EnumDescriptor(
  name='Context',
  full_name='protobuf.message.input.GameEvents.Context',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_CONTEXT', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SELF', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEAM', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPPONENT', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1278,
  serialized_end=1355,
)
_sym_db.RegisterEnumDescriptor(_GAMEEVENTS_CONTEXT)

_GAMEEVENTS_TEAMCOLOUR = _descriptor.EnumDescriptor(
  name='TeamColour',
  full_name='protobuf.message.input.GameEvents.TeamColour',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_TEAM_COLOUR', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CYAN', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MAGENTA', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1357,
  serialized_end=1417,
)
_sym_db.RegisterEnumDescriptor(_GAMEEVENTS_TEAMCOLOUR)


_GAMEEVENTS_SCORE = _descriptor.Descriptor(
  name='Score',
  full_name='protobuf.message.input.GameEvents.Score',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ownScore', full_name='protobuf.message.input.GameEvents.Score.ownScore', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='opponentScore', full_name='protobuf.message.input.GameEvents.Score.opponentScore', index=1,
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
  serialized_start=137,
  serialized_end=185,
)

_GAMEEVENTS_GOALSCORED = _descriptor.Descriptor(
  name='GoalScored',
  full_name='protobuf.message.input.GameEvents.GoalScored',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.GoalScored.context', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='totalScore', full_name='protobuf.message.input.GameEvents.GoalScored.totalScore', index=1,
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
  serialized_start=187,
  serialized_end=280,
)

_GAMEEVENTS_PENALISATION = _descriptor.Descriptor(
  name='Penalisation',
  full_name='protobuf.message.input.GameEvents.Penalisation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.Penalisation.context', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='robotId', full_name='protobuf.message.input.GameEvents.Penalisation.robotId', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ends', full_name='protobuf.message.input.GameEvents.Penalisation.ends', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reason', full_name='protobuf.message.input.GameEvents.Penalisation.reason', index=3,
      number=4, type=14, cpp_type=8, label=1,
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
  serialized_start=283,
  serialized_end=487,
)

_GAMEEVENTS_UNPENALISATION = _descriptor.Descriptor(
  name='Unpenalisation',
  full_name='protobuf.message.input.GameEvents.Unpenalisation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.Unpenalisation.context', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='robotId', full_name='protobuf.message.input.GameEvents.Unpenalisation.robotId', index=1,
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
  serialized_start=489,
  serialized_end=583,
)

_GAMEEVENTS_COACHMESSAGE = _descriptor.Descriptor(
  name='CoachMessage',
  full_name='protobuf.message.input.GameEvents.CoachMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.CoachMessage.context', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='protobuf.message.input.GameEvents.CoachMessage.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=585,
  serialized_end=677,
)

_GAMEEVENTS_HALFTIME = _descriptor.Descriptor(
  name='HalfTime',
  full_name='protobuf.message.input.GameEvents.HalfTime',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='firstHalf', full_name='protobuf.message.input.GameEvents.HalfTime.firstHalf', index=0,
      number=1, type=8, cpp_type=7, label=1,
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
  serialized_start=679,
  serialized_end=708,
)

_GAMEEVENTS_BALLKICKEDOUT = _descriptor.Descriptor(
  name='BallKickedOut',
  full_name='protobuf.message.input.GameEvents.BallKickedOut',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.BallKickedOut.context', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time', full_name='protobuf.message.input.GameEvents.BallKickedOut.time', index=1,
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
  serialized_start=710,
  serialized_end=828,
)

_GAMEEVENTS_KICKOFFTEAM = _descriptor.Descriptor(
  name='KickOffTeam',
  full_name='protobuf.message.input.GameEvents.KickOffTeam',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='protobuf.message.input.GameEvents.KickOffTeam.context', index=0,
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
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=830,
  serialized_end=904,
)

_GAMEEVENTS_GAMEPHASE = _descriptor.Descriptor(
  name='GamePhase',
  full_name='protobuf.message.input.GameEvents.GamePhase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phase', full_name='protobuf.message.input.GameEvents.GamePhase.phase', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='readyTime', full_name='protobuf.message.input.GameEvents.GamePhase.readyTime', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endHalf', full_name='protobuf.message.input.GameEvents.GamePhase.endHalf', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ballFree', full_name='protobuf.message.input.GameEvents.GamePhase.ballFree', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ends', full_name='protobuf.message.input.GameEvents.GamePhase.ends', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nextHalf', full_name='protobuf.message.input.GameEvents.GamePhase.nextHalf', index=5,
      number=6, type=11, cpp_type=10, label=1,
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
  serialized_start=907,
  serialized_end=1205,
)

_GAMEEVENTS_GAMEMODE = _descriptor.Descriptor(
  name='GameMode',
  full_name='protobuf.message.input.GameEvents.GameMode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mode', full_name='protobuf.message.input.GameEvents.GameMode.mode', index=0,
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
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1207,
  serialized_end=1276,
)

_GAMEEVENTS = _descriptor.Descriptor(
  name='GameEvents',
  full_name='protobuf.message.input.GameEvents',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[_GAMEEVENTS_SCORE, _GAMEEVENTS_GOALSCORED, _GAMEEVENTS_PENALISATION, _GAMEEVENTS_UNPENALISATION, _GAMEEVENTS_COACHMESSAGE, _GAMEEVENTS_HALFTIME, _GAMEEVENTS_BALLKICKEDOUT, _GAMEEVENTS_KICKOFFTEAM, _GAMEEVENTS_GAMEPHASE, _GAMEEVENTS_GAMEMODE, ],
  enum_types=[
    _GAMEEVENTS_CONTEXT,
    _GAMEEVENTS_TEAMCOLOUR,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=123,
  serialized_end=1417,
)

_GAMEEVENTS_SCORE.containing_type = _GAMEEVENTS
_GAMEEVENTS_GOALSCORED.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_GOALSCORED.containing_type = _GAMEEVENTS
_GAMEEVENTS_PENALISATION.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_PENALISATION.fields_by_name['ends'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_PENALISATION.fields_by_name['reason'].enum_type = message_dot_input_dot_GameState__pb2._GAMESTATE_DATA_PENALTYREASON
_GAMEEVENTS_PENALISATION.containing_type = _GAMEEVENTS
_GAMEEVENTS_UNPENALISATION.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_UNPENALISATION.containing_type = _GAMEEVENTS
_GAMEEVENTS_COACHMESSAGE.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_COACHMESSAGE.containing_type = _GAMEEVENTS
_GAMEEVENTS_HALFTIME.containing_type = _GAMEEVENTS
_GAMEEVENTS_BALLKICKEDOUT.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_BALLKICKEDOUT.fields_by_name['time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_BALLKICKEDOUT.containing_type = _GAMEEVENTS
_GAMEEVENTS_KICKOFFTEAM.fields_by_name['context'].enum_type = _GAMEEVENTS_CONTEXT
_GAMEEVENTS_KICKOFFTEAM.containing_type = _GAMEEVENTS
_GAMEEVENTS_GAMEPHASE.fields_by_name['phase'].enum_type = message_dot_input_dot_GameState__pb2._GAMESTATE_DATA_PHASE
_GAMEEVENTS_GAMEPHASE.fields_by_name['readyTime'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_GAMEPHASE.fields_by_name['endHalf'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_GAMEPHASE.fields_by_name['ballFree'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_GAMEPHASE.fields_by_name['ends'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_GAMEPHASE.fields_by_name['nextHalf'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_GAMEEVENTS_GAMEPHASE.containing_type = _GAMEEVENTS
_GAMEEVENTS_GAMEMODE.fields_by_name['mode'].enum_type = message_dot_input_dot_GameState__pb2._GAMESTATE_DATA_MODE
_GAMEEVENTS_GAMEMODE.containing_type = _GAMEEVENTS
_GAMEEVENTS_CONTEXT.containing_type = _GAMEEVENTS
_GAMEEVENTS_TEAMCOLOUR.containing_type = _GAMEEVENTS
DESCRIPTOR.message_types_by_name['GameEvents'] = _GAMEEVENTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GameEvents = _reflection.GeneratedProtocolMessageType('GameEvents', (_message.Message,), dict(

  Score = _reflection.GeneratedProtocolMessageType('Score', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_SCORE,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.Score)
    ))
  ,

  GoalScored = _reflection.GeneratedProtocolMessageType('GoalScored', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_GOALSCORED,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.GoalScored)
    ))
  ,

  Penalisation = _reflection.GeneratedProtocolMessageType('Penalisation', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_PENALISATION,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.Penalisation)
    ))
  ,

  Unpenalisation = _reflection.GeneratedProtocolMessageType('Unpenalisation', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_UNPENALISATION,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.Unpenalisation)
    ))
  ,

  CoachMessage = _reflection.GeneratedProtocolMessageType('CoachMessage', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_COACHMESSAGE,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.CoachMessage)
    ))
  ,

  HalfTime = _reflection.GeneratedProtocolMessageType('HalfTime', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_HALFTIME,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.HalfTime)
    ))
  ,

  BallKickedOut = _reflection.GeneratedProtocolMessageType('BallKickedOut', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_BALLKICKEDOUT,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.BallKickedOut)
    ))
  ,

  KickOffTeam = _reflection.GeneratedProtocolMessageType('KickOffTeam', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_KICKOFFTEAM,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.KickOffTeam)
    ))
  ,

  GamePhase = _reflection.GeneratedProtocolMessageType('GamePhase', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_GAMEPHASE,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.GamePhase)
    ))
  ,

  GameMode = _reflection.GeneratedProtocolMessageType('GameMode', (_message.Message,), dict(
    DESCRIPTOR = _GAMEEVENTS_GAMEMODE,
    __module__ = 'message.input.GameEvents_pb2'
    # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents.GameMode)
    ))
  ,
  DESCRIPTOR = _GAMEEVENTS,
  __module__ = 'message.input.GameEvents_pb2'
  # @@protoc_insertion_point(class_scope:protobuf.message.input.GameEvents)
  ))
_sym_db.RegisterMessage(GameEvents)
_sym_db.RegisterMessage(GameEvents.Score)
_sym_db.RegisterMessage(GameEvents.GoalScored)
_sym_db.RegisterMessage(GameEvents.Penalisation)
_sym_db.RegisterMessage(GameEvents.Unpenalisation)
_sym_db.RegisterMessage(GameEvents.CoachMessage)
_sym_db.RegisterMessage(GameEvents.HalfTime)
_sym_db.RegisterMessage(GameEvents.BallKickedOut)
_sym_db.RegisterMessage(GameEvents.KickOffTeam)
_sym_db.RegisterMessage(GameEvents.GamePhase)
_sym_db.RegisterMessage(GameEvents.GameMode)


# @@protoc_insertion_point(module_scope)
