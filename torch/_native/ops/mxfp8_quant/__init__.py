from .. import tu

if tu.runtime_available():
    from .mxfp8_triton import register_to_dispatcher
    tu.register_op(register_to_dispatcher)
