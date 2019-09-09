# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gabriel.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gabriel.proto',
  package='gabriel',
  syntax='proto3',
  serialized_options=_b('\n\032edu.cmu.cs.gabriel.networkB\006Protos'),
    serialized_pb=_b(
        '\n\rgabriel.proto\x12\x07gabriel\"\xa2\x01\n\x05Input\x12\x10\n\x08'
        '\x66rame_id\x18\x01 \x01(\x04\x12!\n\x04type\x18\x02 \x01('
        '\x0e\x32\x13.gabriel.Input.Type\x12\x0f\n\x07payload\x18\x03 \x01('
        '\x0c\x12\r\n\x05style\x18\x04 \x01('
        '\t\x12\x10\n\x08\x66ilename\x18\x05 \x01('
        '\t\"2\n\x04Type\x12\t\n\x05IMAGE\x10\x00\x12\t\n\x05VIDEO\x10\x01'
        '\x12\t\n\x05\x41UDIO\x10\x02\x12\t\n\x05\x41\x43\x43\x45L\x10\x03'
        '\"\xac\x02\n\x06Output\x12\x10\n\x08\x66rame_id\x18\x01 \x01('
        '\x04\x12&\n\x06status\x18\x02 \x01('
        '\x0e\x32\x16.gabriel.Output.Status\x12\'\n\x07results\x18\x03 \x03('
        '\x0b\x32\x16.gabriel.Output.Result\x1a\x43\n\x06Result\x12('
        '\n\x04type\x18\x01 \x01('
        '\x0e\x32\x1a.gabriel.Output.ResultType\x12\x0f\n\x07payload\x18\x02 '
        '\x01(\x0c\"7\n\x06Status\x12\x0b\n\x07SUCCESS\x10\x00\x12\x10\n\x0c'
        '\x45NGINE_ERROR\x10\x01\x12\x0e\n\nTASK_ERROR\x10\x02\"A\n'
        '\nResultType\x12\x08\n\x04\x41NIM\x10\x00\x12\t\n\x05VIDEO\x10\x01'
        '\x12\t\n\x05IMAGE\x10\x02\x12\t\n\x05\x41UDIO\x10\x03\x12\x08\n'
        '\x04TEXT\x10\x04\x42$\n\x1a\x65\x64u.cmu.cs.gabriel.networkB'
        '\x06Protosb\x06proto3')
)



_INPUT_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='gabriel.Input.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IMAGE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VIDEO', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AUDIO', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACCEL', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=139,
  serialized_end=189,
)
_sym_db.RegisterEnumDescriptor(_INPUT_TYPE)

_OUTPUT_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='gabriel.Output.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
        name='ENGINE_ERROR', index=1, number=1,
        serialized_options=None,
        type=None),
      _descriptor.EnumValueDescriptor(
          name='TASK_ERROR', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=370,
    serialized_end=425,
)
_sym_db.RegisterEnumDescriptor(_OUTPUT_STATUS)

_OUTPUT_RESULTTYPE = _descriptor.EnumDescriptor(
  name='ResultType',
  full_name='gabriel.Output.ResultType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ANIM', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VIDEO', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMAGE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AUDIO', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEXT', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
    serialized_start=427,
    serialized_end=492,
)
_sym_db.RegisterEnumDescriptor(_OUTPUT_RESULTTYPE)


_INPUT = _descriptor.Descriptor(
  name='Input',
  full_name='gabriel.Input',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='gabriel.Input.frame_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='gabriel.Input.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='gabriel.Input.payload', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='style', full_name='gabriel.Input.style', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filename', full_name='gabriel.Input.filename', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _INPUT_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=189,
)


_OUTPUT_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='gabriel.Output.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='gabriel.Output.Result.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='gabriel.Output.Result.payload', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=301,
  serialized_end=368,
)

_OUTPUT = _descriptor.Descriptor(
  name='Output',
  full_name='gabriel.Output',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='gabriel.Output.frame_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='gabriel.Output.status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='results', full_name='gabriel.Output.results', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_OUTPUT_RESULT, ],
  enum_types=[
    _OUTPUT_STATUS,
    _OUTPUT_RESULTTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
    serialized_end=492,
)

_INPUT.fields_by_name['type'].enum_type = _INPUT_TYPE
_INPUT_TYPE.containing_type = _INPUT
_OUTPUT_RESULT.fields_by_name['type'].enum_type = _OUTPUT_RESULTTYPE
_OUTPUT_RESULT.containing_type = _OUTPUT
_OUTPUT.fields_by_name['status'].enum_type = _OUTPUT_STATUS
_OUTPUT.fields_by_name['results'].message_type = _OUTPUT_RESULT
_OUTPUT_STATUS.containing_type = _OUTPUT
_OUTPUT_RESULTTYPE.containing_type = _OUTPUT
DESCRIPTOR.message_types_by_name['Input'] = _INPUT
DESCRIPTOR.message_types_by_name['Output'] = _OUTPUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), dict(
  DESCRIPTOR = _INPUT,
  __module__ = 'gabriel_pb2'
  # @@protoc_insertion_point(class_scope:gabriel.Input)
  ))
_sym_db.RegisterMessage(Input)

Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), dict(

  Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), dict(
    DESCRIPTOR = _OUTPUT_RESULT,
    __module__ = 'gabriel_pb2'
    # @@protoc_insertion_point(class_scope:gabriel.Output.Result)
    ))
  ,
  DESCRIPTOR = _OUTPUT,
  __module__ = 'gabriel_pb2'
  # @@protoc_insertion_point(class_scope:gabriel.Output)
  ))
_sym_db.RegisterMessage(Output)
_sym_db.RegisterMessage(Output.Result)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
