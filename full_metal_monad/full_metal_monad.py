import jmespath;from functools import reduce
from types import FunctionType
from rich.console import Console
from returns.maybe import Maybe, Some, Nothing
import builtins
from multipledispatch import dispatch
from collections.abc import Iterable, Mapping, Sequence

NoneType = type(None)
# console = Console()

class MetalMonad:
    def __init__(self, data, safe=None):
        self.data = Some(data)
        self._val = False
        self._inplace = False
        self.safe = safe

    def __getattr__(self, name):
        return __.FunctionRunner(self, name)

    def __repr__(self):
        return f"MetalMonad({self.data}, safe:{self.safe})"

    @property
    def __(self):
        if isinstance(self.data.unwrap(),type(Nothing)):
            return self.safe
        else:
            return self.data.value_or(self.safe)

    @property
    def val(self):
        self._val = True
        return self

    @property
    def inplace(self):
        self._inplace = True
        return self

    def extract_data(self):
        if isinstance(self.data, Some):
            if self.safe == None:
                data=self.data.unwrap()
            else:
                if isinstance(self.data.unwrap(), type(Nothing)):
                    data=self.safe
                else:
                    data=self.data.value_or(self.safe)
        else:
            data=self.data
        return data


    def get(self,fields,safe=None,pattern=None):
        # try:
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

    def map(self, func):
        data=self.extract_data()
        # if isinstance(data[0], list) or isinstance(data[0], dict):
        #     return [__(i).get(fields).__ for i in data]
        # else:
        return __([func(i) for i in data])

    def remove(self, fields):
        data=self.extract_data()            
        if type(fields) == str:
            fields = [fields]
        if isinstance(data, dict):
            if self._inplace:
                [data.pop(key, None) for key in fields]
            else:
                data={k:v for k,v in data.items() if k not in fields}
        if isinstance(data, list):
            if isinstance(data[0], dict):
                data=[{k:v for k,v in i.items() if k not in fields} for i in data]
            if isinstance(data[0], str):
                data=[k for k in data if k not in fields]
        
        return __(data)

    def single_value_or(self, message):
        if not len(self) == 1:
            raise ValueError(message)
        return True

    def single_or(self, message):
        if not len(self) == 1:
            raise ValueError(message)
        return True

    def nested_to_flat(self, keys_to_capture,pattern='references.hasChildren',acc_id=None,delete_pattern=False,names={},temp_keep_format=False):
        """# pythonOntology.flat(keys_to_capture) WISHED SYNTAX
        I fucking almost did it
        a = _(newHierarchy).nested_to_flat(keys_to_capture='*',pattern='children')   
        """

        pattern_split = pattern.split('.')
        if keys_to_capture == '*':
            keys_to_capture = list(set(list(self.data.keys())) - set([pattern[0]]))
        if isinstance(keys_to_capture, str):
            keys_to_capture = [keys_to_capture]

        def rget(data, keys):
            def safe_get(acc, key):
                if isinstance(acc, list):
                    return [safe_get(item, key) for item in acc]
                return acc.get(key, None) if isinstance(acc, dict) else None
            return reduce(safe_get, keys, data)

        def get_level(data, keys_to_capture):
            level_data = [rget(data, key.split('.')) for key in keys_to_capture]
            zipped = [(x, y) for x, y in zip(keys_to_show, level_data) if y is not None]
            return dict(zipped)

        def all_levels(data, keys_to_capture, acc_id,acc_id_value):
            if len(data) == 0:
                return []
            r = get_level(data, keys_to_capture)
            results = [r] if len(r) > 0 else []
            if acc_id:
                acc_id_value='.'.join([acc_id_value, results[0][acc_id]])
                results[0]['acc_id']=acc_id_value
            children = rget(data, pattern_split)
            if isinstance(children, list):
                for child in children:
                    if child:
                        res = all_levels(child,keys_to_capture,acc_id,acc_id_value)
                        if len(res) > 0 and len(res[0]) > 0:
                            results.extend(res)
            return results

        def keys_to_show_strategy(keys_to_capture, names):
            keys_to_capture = [names.get(item, item) for item in keys_to_capture]
            keys_to_capture_end = [i.split('.')[-1] for i in keys_to_capture]
            all_distinct = len(keys_to_capture_end) == len(set(keys_to_capture_end))
            if all_distinct:
                keys_to_show = keys_to_capture_end
            else:
                keys_to_show = keys_to_capture
            return keys_to_show

        keys_to_show = keys_to_show_strategy(keys_to_capture, names)

        res = all_levels(self.data.unwrap(), keys_to_capture, acc_id, '')
        return __(res)

    def flatten(self):
        data=self.extract_data()        
        @dispatch(object)
        def _flatten(x):
            return [x]
        
        @dispatch(Iterable)
        def _flatten(L):
            return sum([_flatten(x) for x in L], [])

        @dispatch(str)
        def _flatten(s):
            return [s]

        return _flatten(data)
        
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



def safe_jmes_search(query, data) -> Maybe:
    result = jmespath.search(query, data)
    if result is not None:
        return Some(result)
    return Nothing


__ = MetalMonad