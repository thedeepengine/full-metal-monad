import jmespath;import builtins
from functools import reduce;from multipledispatch import dispatch
from types import FunctionType;from rich.console import Console
from returns.maybe import Maybe, Some, Nothing
from collections.abc import Iterable, Mapping, Sequence
NoneType = type(None)

class MetalMonad:
    def __init__(self, data, safe=None):
        self.data = Some(data)
        self._val = False
        self.safe = safe

    def __getattr__(self, name):
        return __.FunctionRunner(self, name)

    def __repr__(self):
        return f"MetalMonad({self.data}, safe:{self.safe})"

    @property
    def __(self):
        return self.safe if isinstance(self.data.unwrap(),type(Nothing)) else self.data.value_or(self.safe)

    @property
    def val(self):
        self._val = True
        return self

    def extract_data(self):
        if isinstance(self.data, Some):
            if self.safe == None:
                data=self.data.unwrap()
            else:
                data=self.data.__
                if isinstance(self.data.unwrap(), type(Nothing)):
                    data=self.safe
                else:
                    data=self.data.value_or(self.safe)
        else:
            data=self.data
        return data

    def get(self,fields,safe=None,pattern=None):
        data=self.extract_data()
        self.safe = safe

        @dispatch(Mapping,FunctionType)
        def get(data, func):
            return {k:v for k,v in data.items() if func((k,v))}

        @dispatch(Mapping,list)
        def get(data, field):
            return {k:v for k,v in data.items() if k in field}

        @dispatch(Mapping, str)
        def get(data, field):
            if '.' in field:
                return jmespath.search(field, data)
            else:
                return data.get(field, Nothing)

        @dispatch(dict, str, str)
        def get(data,fields,pattern):
            if pattern:
                acc=[]
                def recurse(data,pattern,fields):
                    if (to_add:=jmespath.search(fields,data)): 
                        acc.append(to_add)
                    if (sub:=jmespath.search(pattern,data)): 
                        for i in sub: recurse(i,pattern,fields)
                recurse(data,pattern,fields)
                return acc

        @dispatch(Sequence,FunctionType)
        def get(data, func):
            return [y for y in data if func(y)]
            
        @dispatch(Sequence,list)
        def get(data, field):
            if all([isinstance(i, bool) for i in field]):
                return [v for v in data if field]
            elif all([isinstance(i, int) for i in field]):
                return [data[v] for v in field]
            else:
                return [v for v in data if v in field]

        @dispatch(Sequence, (str,int,tuple))
        def get(data,fields):
            if isinstance(data[0], list) or isinstance(data[0], dict):
                return [__(i).get(fields).__ for i in data]
            else:
                if isinstance(fields, tuple):
                    fields=slice(fields[0], fields[1])
                if isinstance(fields, int):
                    return data[fields]                

        if pattern != None:
            return __(get(data, fields,pattern), safe)
        else:
            return __(get(data, fields), safe)

    def apply(self, func, fields=None):
        data=self.extract_data()
        @dispatch(list, FunctionType, [(str,list,NoneType)])
        def _apply(data, func, fields):
            if fields != None:
                if type(fields) == str: fields = [fields]
                if all([isinstance(i, dict) for i in data]):
                    res = [{k: (func(v) if k in fields else v) for k, v in i.items()} for i in self.data]
                    return [{k:v for k, v in row.items() if k in fields} for row in res]
            else:
                return [func(i) for i in data]

        return _apply(data,func,fields)

    class FunctionRunner:
        def __init__(self, owner, func):
            self.owner = owner
            self._val = owner._val
            self.func_str = func

        def __call__(self, *args, **kwargs):
            new_data = []
            if isinstance(self.owner.data, Some):
                method = getattr(self.owner.data, self.func_str)
                value = method(*args, **kwargs)
                return MetalMonad(value)
            if hasattr(self.owner.data[0], self.func_str):
                for item in self.owner.data:
                    method = getattr(item, self.func_str)
                    value = method(*args, **kwargs)
                    if self._val: # TODO value should be boolean when self._val=true
                        if value:
                            new_data.append(item)
                    else:
                        new_data.append(value)
            else:
                self.func = getattr(builtins, self.func_str)
                if self.func_str == 'filter':
                    new_data.append(self.func(*args,self.owner.data))
                else:
                    for item in self.owner.data:
                        value=self.func(item, *args, **kwargs)
                        if self._val:
                            if value:
                                new_data.append(item)
                        else:
                            new_data.append(value)
            return MetalMonad(new_data)

__ = MetalMonad