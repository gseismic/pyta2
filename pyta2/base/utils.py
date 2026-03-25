from typing import Any, Dict, List, Optional, Type
from .indicator import rIndicator
from ..utils.vector import VectorTable

def get_outputs(outputs: VectorTable, return_type='dict', reverse=False):
    return_type = return_type.lower()
    if return_type == 'tuple':
        # (x, y, z)
        result = tuple(outputs.to_dict(reverse=reverse).values())
        if len(result) == 1:
            return result[0]
        return result
    elif return_type == 'dict':
        # {'x': x, 'y': y, 'z': z}
        return outputs.to_dict(reverse=reverse)
    elif return_type == 'list':
        # [{'x': x, 'y': y, 'z': z}, ...]
        return outputs.to_list(reverse=reverse)
    elif return_type == 'pl.dataframe':
        import polars as pl
        # pl.DataFrame([{'x': x, 'y': y, 'z': z}, ...])
        return pl.DataFrame(outputs.to_list(reverse=reverse), infer_schema_length=None)
    elif return_type in ['pd.dataframe', 'dataframe']:
        import pandas as pd
        # pd.DataFrame([{'x': x, 'y': y, 'z': z}, ...])
        return pd.DataFrame(outputs.to_list(reverse=reverse))
    else:
        raise Exception(f'Not Supported `{return_type=}`')
        
def forward_rolling_apply(num: int, obj_cls: Type[rIndicator], param_args: List[Any] = [],
                          param_kwargs: Dict[str, Any] = {},
                          input_args: List[Any] = [], input_kwargs: Dict[str, Any] = {},
                          doroll_input_args: Optional[List[bool]] = None,
                          doroll_input_kwargs: Optional[Dict[str, bool]] = None,
                          return_type: str = 'tuple', return_meta_info: bool = False):
    '''
    apply rolling forward
    '''
    assert 'buffer_size' not in param_kwargs, 'buffer_size is not allowed in param_kwargs'
    assert doroll_input_args is None or len(doroll_input_args) == len(input_args), \
        f'doroll_input_args must be None or have the same length as input_args, got len(doroll_input_args)={len(doroll_input_args) if doroll_input_args else None}, len(input_args)={len(input_args)}'
    if num < 1:
        raise ValueError(f'num must be greater than 0, got {num}')
    param_kwargs = {**param_kwargs, 'return_dict': True}
    obj = obj_cls(*param_args, **param_kwargs)
    if hasattr(obj, 'reset'):
        obj.reset()

    output_table = VectorTable()
    if doroll_input_args is None:
        doroll_input_args = [True if arg is not None else False for arg in input_args]
    if doroll_input_kwargs is None:
        doroll_input_kwargs = {key: True if val is not None else False for key, val in input_kwargs.items()}

    def _slice_input(arg, i, do_roll):
        if not do_roll:
            return arg
        if hasattr(arg, 'values'):
            return arg.values[:i+1]
        return arg[:i+1]

    for i in range(num):
        sliced_args = [_slice_input(arg, i, vtype) for vtype, arg in zip(doroll_input_args, input_args)]
        sliced_kwargs = {key: _slice_input(val, i, doroll_input_kwargs.get(key, True)) for key, val in input_kwargs.items()}
        output = obj.rolling(*sliced_args, **sliced_kwargs)
        output_table.append(output)

    if return_meta_info:
        return get_outputs(output_table, return_type=return_type), obj.meta_info
    else:
        return get_outputs(output_table, return_type=return_type)
