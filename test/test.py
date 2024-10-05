import unittest
from typing import FunctionType
import jmespath

from full_metal_monad import MetalMonad

def tag(*tags):
    def decorator(func):
        setattr(func, "tags", tags)
        return func
    return decorator

class TestMetalMonad(unittest.TestCase):
    def setUp(self):
        # Initialize test data for various tests
        self.monad_dict = MetalMonad({"name": "Alice", "age": 30, "city": "New York"})
        self.monad_list = MetalMonad([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
        self.monad_numbers = MetalMonad([10, 20, 30, 40, 50])

    def test_single_field_query(self):
        # Testing retrieval by single field name
        self.assertEqual(self.monad_dict.get("name"), "Alice")
        self.assertEqual(self.monad_dict.get("age"), 30)

    def test_multiple_field_query(self):
        # Testing retrieval by multiple field name
        self.assertEqual(self.monad_dict.get("name", "age"), "Alice")

    def test_function_query(self):
        # Define function for filtering
        def age_check(item):
            key, value = item
            return key == "age" and value > 20

        # Testing with functions
        self.assertEqual(self.monad_dict.get(age_check), {"age": 30})
        self.assertEqual(self.monad_dict.get(lambda item: item[0] == "name"), {"name": "Alice"})
        self.assertEqual(self.monad_dict.get(lambda item: item[1] == "New York"), {"city": "New York"})

    @tag('data_type:dict', 'fields:string.single_value')
    def test_a6fj65(self):
        # Testing with list of field names
        self.assertEqual(self.monad_dict.get(["name", "age"]), {"name": "Alice", "age": 30})
        self.assertEqual(self.monad_dict.get(["city"]), {"city": "New York"})
        self.assertEqual(self.monad_dict.get(["age", "city"]), {"age": 30, "city": "New York"})

    def test_function_query_on_list(self):
        # Define function for filtering list of dicts
        def age_over_25(person):
            return person["age"] > 25

        # Testing with functions on list of dictionaries
        self.assertEqual(self.monad_list.get(age_over_25), [{"name": "Alice", "age": 30}])
        self.assertEqual(self.monad_list.get(lambda x: x["name"] == "Alice"), [{"name": "Alice", "age": 30}])
        self.assertEqual(self.monad_list.get(lambda x: x["age"] < 30), [{"name": "Bob", "age": 25}])

    def test_dictionary_query_on_list(self):
        # Testing dictionary-based field selection on list of dicts
        self.assertEqual(self.monad_list.get({"name", "age"}), [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
        self.assertEqual(self.monad_list.get({"city"}), ["New York", "LA"])
        self.assertEqual(self.monad_list.get({"age"}), [30, 25])

    def test_list_of_indices_query(self):
        # Testing list of indices on list of numbers
        self.assertEqual(self.monad_numbers.get([1, 3]), [20, 40])
        self.assertEqual(self.monad_numbers.get([0, 2, 4]), [10, 30, 50])
        self.assertEqual(self.monad_numbers.get([4]), [50])



__(['aa', 'aaab', 'aaabb', 'ffff']).startswith('a')


__({'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}).get('key1')
__({'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}).get(lambda t: t[1] == 'aaa')
__({'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}).get(lambda t: t[0] == 'key2')
__({'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}).get(['key1', 'key3'])
__({'key1': 'aaa', 'key2': 'bbb', 'key3': {'nested_key4': 'lll'}}).get('key3.nested_key4')


dict(filter((lambda x: x == 'key1'), {'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}.items()))

list(filter(lambda item: item == 'key1', {'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}))


# Examples:
my_list = [1, 2, 3, 4, 5]
filtered_list = generic_filter(my_list, lambda x: x > 3)
print(filtered_list)  # Output: [4, 5]

my_dict = {'a': 1, 'b': 2, 'c': 3}
filtered_dict_keys = generic_filter(my_dict, lambda k: my_dict[k] > 1)
print(filtered_dict_keys)  # Output: ['b', 'c']


if __name__ == '__main__':
    unittest.main()

# from full_metal_monad import MetalMonad
# MetalMonad({'name': 'test_444'}).get('name')
# MetalMonad({'name': 'test_444'}).data.items()



from multipledispatch import dispatch

@dispatch((int,float), int)
def add(x, y):
    return x + y

__([1,2,3,4,5]).apply(lambda x: x+2)


__(['AAA', 'BBB', 'CCCC']).lower()

__(['AAA', 'BBB', ' CCCC ']).strip().lower().apply(lambda x: 'prefix_'+x)
result = ['prefix_' + item.strip().lower() for item in array]


@dispatch(Iterable)
def flatten(L):
    return sum([flatten(x) for x in L], [])

@dispatch(object)
def flatten(x):
    return [x]





@dispatch(Iterable,FunctionType)
def get(x, func):
    if isinstance(x, Mapping):
        x =  x.items()
    return __([y for y in x if func(y)])

@dispatch(Iterable)
def get(x):
    if isinstance(x, Mapping):
        x =  x.items()
    return __([y for y in x if func(y)])

@dispatch(list, int) 
def get(fields):
    if 0 <= fields < len(data):
        return data[fields]
    else:
        return None

get([1,2,3,4])



data=[1,2,3,4,5,6,2,2,3,4]
# one index
data[0]
__(data).get(0).__

# multiple indices
indices=[1,2,3]
[data[i] for i in indices]
[data[i] for i in indices]
__(data).get([2,4,5]).__

# slices
slice_i=slice(3,4)
data[slice(3,5)]


dict(filter(fields, data.items()))

dict(filter(lambda x: x[0] == 'key1', {'key1': '33', 'key2': 'ooo'}.items()))




@dispatch(type(Nothing), object)
def get(data, fields):
    # print('aaaaa')
    return None


def debug_dispatch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Called function: {func.__name__} with args: {args} and kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper




[x for x in iterable1 if condition1(x) for y in iterable2 if condition2(y)]


__([[2,3,4],2,3,4]).get(0).get().get(lambda x: [x[0], x[1]])


__([[2,3,4],2,3,4])._()

[item for item, keep in zip(data, filter_mask) if keep]

t=[[1,2,3,4,5,6],
    [[1,2,3], [4,5,6,7],[8,9]],
   [{'key1': 1, 'key2': 2},{'key1': 3, 'key2': 4}]]


__(t[0]).get([True,False,False,True,True,False]).__

__(t[1]).get(0).__

__(t[2]).get('key1').__

__(data).get(0)














### dict
data={'key1': '33', 
      'key2': 'ooo', 
      'key3': 'kkkk', 
      'nkey1': 'ggg', 
      'nkey2': 'll',
      'key6': {'nested1':'aaa', 'nested2': {'nested21': 'aaa'}}}

# get a single key out of a dict
key = 'key1'
# native
data[key]
# mm
__(data).get('key1').__


__(data).get('keyaa1', 'kkk').__

# get a list of keys out of a dict
list_keys=['key1', 'key2']
# native
{k:v for k,v in data.items() if k in list_keys}
# mm
__(data).get(['key1', 'key2']).__

# get items based on conditional boolean function for the key
func=lambda x: x[0].startswith('k')
# native
{k:v for k,v in data.items() if func(k)}
# mm
__(data).get(func).__

# get nested items 
__(data).get('key6.nested1').__



l = [data,data,data]

__(l).get('key1').__
__(l).get(['key1', 'key2'])
__(l).get(func).__
__(l).get('key6.nested1').__


__(l).get(lambda x: x[1].startswith('o') if not isinstance(x[1], dict) else 'k').__


### list 
data=[1,2,3,4,5,6]

# get by index
# native 
data[2]
# mm
__(data).get(2).__
# safe mm
__(data).get(99, safe=0).__

# get multiple indices
indices=[2,3,4]
# native
[data[i] for i in indices]
# mm
__(data).get(indices).__

# get a slice
s = slice(0,3)
# native
data[s]
# mm
__(data).get((0,3)).__

# get with boolean function
func = lambda x: x > 2
# native
[i for i in data if func(i)]
# mm
__(data).get(func).__

# get values
values = [2,3,4]
# native
[i for i in data if i in values]



__(data).i.get([2,3])




# list of dict
data=[{'key1': 1, 'key2': 2},{'key1': 3, 'key2': 4},{'key1': 5, 'key2': 6,'key3':9}]

# native
[d['key1'] for d in data]
# mm
__(data).get('key1').__




[__(i).get(['key1', 'key2']).__ for i in data]

[__(i).get(['key3']).__ for i in data]


__(data).get(['key1','key2']).__
__(data).get('key1').__

__(data).get(['key1', 'key2']).__

__(data).get('key3').__

__(data).get(lambda x: x[1]>4).__


__({'key1': 5, 'key2': 6,'key3':9}).get(['key3', 'key2']).__


[__(i).get(['key3']).__ for i in data]
[__(i).get('key3').__ for i in data]

key3_values = [d['key3'] for d in data if 'key3' in d]


__({'key1': 1, 'key2': 2, 'key3': 9}).get(['key1','key2']).__


# indexing by index or by the actual value

data = [1,2,3,4,5,6]
[i for i in data if i in [1,2,3]]
[data[i] for i in [1,2,3]]







@dispatch(list, (int,tuple))
def get(data,fields):
    if wise or not isinstance(data[0], list):
        if isinstance(fields, tuple):
            fields=slice(fields[0], fields[1])
        if isinstance(fields, int):
            return  __(data[fields], safe)
    elif isinstance(data[0], list):
        return 

        
    get(data)


# list of list of int
data=[[1,2,3], [4,5,6], [7,8,9]]

# select a given index
index=2
__(data).get(index).__

# by default, if the input is an array of array, the filter is applied wisely, for each element
# you can force it to be applied globally using the wise property
__(data).wise.get(2).__

func = lambda x: x>2
__(data).get(lambda x: x>2).__

[[j for j in i if func(j)] for i in data]


__(data).get([1,2]).__

__([1,2,3]).get([1,2]).__
__([1,2,3]).get(0).__

__([[1,2,3], [4,5,6], [7,8,9]]).get([1,2]).__



[[i[index]] for i in data]

__(data).wise.get(2).__

[i]

# 

__(data).get(lambda x: x[0]).__



ont_uuid_map = __(ont_uuid_map).nested_to_flat(['uuid', 'properties.name'])


safe=None


data=[[1,2,3], [4,5,6], [7,8,9]]
getA(data, 0)

@dispatch(dict, str)
def getA(data, field):
    if '.' in field:
        return __(jmespath.search(field, data),safe)
    else:
        return __(data.get(field, Nothing),safe)

@dispatch(list, (str,int)) 
def getA(data,field):
    return __(data[field]).__

@dispatch(list, (str,int))
def getA(data,field):

    @dispatch(l)
    def getA(data,field):
        return __([getA(i,field).__ for i in data])

@dispatch(list, FunctionType)
def getA(data,func):
    return __([i for i in data if func(i)])

data=[[1,2,3], [4,5,6], [7,8,9]]
getA(data, 2)

data=[{'key1': 1, 'key2': 2},{'key1': 3, 'key2': 4},{'key1': 5, 'key2': 6}]
getA(data, 'key1')

data=[[1,2,3], [4,5,6], [7,8,9]]

getA([1,2,3], 0)

getA(data, lambda x: x[0])

__(data).get(lambda x: x[1])


__(data).get(0)

    # return __([jmespath.search(fields, i) for i in data])



__(data).get(99, safe={'k':333}).get('k')









@dispatch(Iterable, (int,str))
def get(data, field):
    return __([get(y, field) for y in data])


@dispatch(Iterable)
def dd(data):
    return __([dd(y, field) for y in data])





@dispatch(Iterable,FunctionType)
def dd(data,field):
    for i in data:
        print(i)
    return [ff(y, field) for y in data]

@dispatch(list,FunctionType)
def ff(data, func):
    print('data', data)
    return __([y for y in data if func(y)])
    
field = lambda x: x>2
dd([[1,2,3],[4,5,6],[7,8,9]],field)



@dispatch(list,FunctionType)
def dd(data,field):
    return [dd(y, field) for y in data]

@dispatch(Iterable,FunctionType)
def dd(data, func):
    return __([y for y in data if func(y)])
    
field = lambda x: x>2
dd([[1,2,3],[4,5,6],[7,8,9]],field)



dd([1,2,3,4,5], field)
get([1,2,3], 2)


#### base example
@dispatch(Iterable)
def f(x):
    return [f(y) for y in x]

@dispatch(int)
def f(x):
    return x+1


f([1,2,3,4,5])


import traceback



array=[1,2,3,4]
array[6]


def format_traceback(exc_info):
    # Extract the traceback from the exception info
    tb = traceback.extract_tb(exc_info[2])
    # Format the traceback entries to be more readable
    readable_traceback = []
    for i, entry in enumerate(tb):
        readable_traceback.append(f"File '{entry.filename}', line {entry.lineno}, in {entry.name}")
        readable_traceback.append(f"    {entry.line}")
    return "\n".join(readable_traceback)


class SimpleException(Exception):
    """Custom exception for cleaner output."""
    def __init__(self, msg=None):
        if msg is None:
            # message = ''.join(traceback_formatter.structured_traceback())
            print(msg)
            # _, _, tb = sys.exc_info()
            # last_call_stack = traceback.extract_tb(tb)[-1]
            # # print('last_call_stack: ', last_call_stack)
            # # print('traceback.extract_tb(tb): ', traceback.extract_tb(tb))
            # print(format_traceback(sys.exc_info()))
            # self.msg = f"Exception in {last_call_stack.filename}, line {last_call_stack.lineno}, in {last_call_stack.name}"
        else:
            self.msg = msg
        super().__init__(self.msg)


def try_except(data,fields,safe):
    try:
        return  __(data[fields], safe)
    except IndexError as e: 
        raise SimpleException('hhhhh')
    except Exception as e:
        raise SimpleException('aaaaa') from None



                        # Only capture and print traceback from the current scope
                        # exc_type, exc_value, exc_traceback = sys.exc_info()
                        # formatted_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback, limit=0)
                        # error_msg = ''.join(formatted_traceback)  # Join the traceback strings into one message
                        # raise Exception(error_msg) from None  # Raise a new exception with limited traceback and no context

                    # except Exception:
                        # console.print_exception(show_locals=True)
                        # console.print(rich.traceback.Traceback(traceback.format_exc()))
                        # rich.traceback.Traceback(traceback)
                        # console.print_exception(traceback.format_exc())
                        # return rich.traceback.Traceback(traceback)
                        # raise Exception
                        # return __(Nothing, safe)
                # return __(data[fields], safe)

@dispatch(Iterable,FunctionType)
def iterate(data,field):
    for i in data:
        print(i)
    return [ff(y, field) for y in data]


a = {'ddd': 1}
a['kkk']

a.llll

[1,2,3,4][100]

__([1,2,3,4]).get(4).get('dddd')

__([1,2,3,4]).get(4).get('hh')

__([1,2,3,4]).get(4).get('hh')


__([1,2,3,4]).get(1).__

array=[{'key': 1}]
__(array).get(0).get('key').get_or_raise()

array=[{'key1': 1}]
array=[{'key1': 1}]
__(array).get(2).get('key2').__


__(array).get(0).get('key2').get_or_raise()

__(array).get(10)
a=__(array).get(10)

.get('key2')


__({'key1': 2}).get('key2')


__(array).get(10, 3)

__([{'key1': 3}]).get(0).get('key2').get(3)


import traceback

def ff(i):
    global dd
    try:
        array=[1,2,3,4]
        return array[i]
    except Exception:
        # print(repr(e))
        # t=traceback.print_exc()
        # print(t)
        # console.log(traceback.format_exc())
        # console.print(traceback.format_exc())
        console.print_exception(show_locals=True)
        dd=traceback
        raise
        # console.print_exception()
        # return e



def ff(i):
    global dd, kk,hh
    try:
        array=[1,2,3,4]
        return array[i]
    except Exception:
        # console.print_exception(show_locals=True)
        # exc_type, exc_value, exc_traceback = sys.exc_info()
        # print(exc_type)
        # a=rich.traceback.Traceback()
        # print('exc_traceback: ', exc_traceback)
        # hh=exc_traceback
        # a.extract(exc_type, exc_value, traceback)
        # console.print(a)
        # print(exc_type)
        dd=traceback
        # kk=exc_traceback
        traceback.print_exc()
        raise

a=ff(100)

rich.traceback.Traceback(dd)

rich.traceback.Traceback.extract(dd)

rich

a=ff(100)

def ff(i):
    try:
        array=[1,2,3,4]
        return array[i]
    except Exception:
        raise

a=ff(100)

def ff(i):
    try:
        array=[1,2,3,4]
        return array[i]
    except Exception:
        print('aaa')

a=ff(100)


def ff(i):
    try:
        array=[1,2,3,4]
        return array[i]
    except:
        raise
a=ff(100)


install(suppress=100)

__(array).get(1).get('key2').__


get([1,2,3], 5)




def ff(i):
    try:
        array = [1, 2, 3, 4]
        return array[i]
    except Exception:
        console.print_exception(max_frames=0)

ff(100)







t=traceback.print_exc()
print(t)
console.log(traceback.format_exc())

# console.print_exception(show_locals=True)
# exc_type, exc_value, exc_traceback = sys.exc_info()
# print(exc_type)
# a=rich.traceback.Traceback()
# print('exc_traceback: ', exc_traceback)
# hh=exc_traceback
# a.extract(exc_type, exc_value, traceback)
# console.print(a)
# print(exc_type)


from rich.traceback import Traceback


@dispatch(list, (int,tuple)) 
def get2(data,fields):
    if isinstance(fields, tuple):
        fields=slice(fields[0], fields[1])
    if isinstance(fields, int):
        # if fields < 0 or fields >= len(data):
        try:
            return  __(data[fields])
        except Exception:
            console.print_exception(show_locals=True)
            # traceback.print_exc()
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # extracted_traceback = Traceback.extract(exc_type, exc_value, exc_traceback)
            # # console.print(Traceback(extracted_traceback))
            # console.print(extracted_traceback)
            # console.print(Traceback(extracted_traceback))
            
            # r=rich.traceback.Traceback.extract(exc_type, exc_value, (Traceback(extracted_traceback)))
            # a=rich.traceback.Traceback(r)
            # console.print(a)
            # raise

@dispatch(list, (int,tuple)) 
def get2(data,fields):
    if isinstance(fields, tuple):
        fields=slice(fields[0], fields[1])
    if isinstance(fields, int):
        # if fields < 0 or fields >= len(data):
        try:
            return  __(data[fields])
        except Exception:
            console.print_exception(show_locals=True)

get2([1,2,3], 5)


try:
    get2([1,2,3], 5)
except Exception:
    print('aa')

def get3(data,fields):
    if isinstance(fields, tuple):
        fields=slice(fields[0], fields[1])
    if isinstance(fields, int):
        # if fields < 0 or fields >= len(data):
        try:
            return  __(data[fields])
        except:
            raise


def gg():
    try:
        get3([1,2,3], 5)
    except Exception:
        # console.print_exception()
        raise Exception

gg()

def ff(i):
    try:
        array=[1,2,3,4]
        return array[i]
    except:
        raise

try:
    a=ff(100)
except:
    raise



def ff(i):
    try:
        array=[1,2,3,4]
        return array[i]
    except Exception as e:
        raise

ff(6)


def function_a():
    function_b()

def function_b():
    function_c()

def function_c():
    # This will raise an error that bubbles up through function_b to function_a
    raise ValueError("An error occurred in function C")

try:
    function_a()
except ValueError as e:
    raise


a=__(array).get(1)


@dispatch(Nothing)
def aaa(data):
    print(data)


a.extract_data()








@dispatch(Iterable,FunctionType)
def get(data, func):
    if isinstance(data, Mapping):
        data = data.items()
    return __([y for y in data if func(y)])


@dispatch(Iterable)
def f(x):
    return [f(y) for y in x]

@dispatch(int)
def f(x):
    return x+1



a([1,2,3,4,5], 2)
a([1,2,3,4,5], (2,4))

[1,2,3,4][slice(2,3)]



__(data).get((0,3)).__

__(data).get(0).get(77, 9).__


__(data).get(0).get(3, {}).__

__(data).get(0).get(1).__

a.data.value_or(8)
a.data.fix(8)

.safe

data=[{'key1': '33', 'key2': 'ooo'}, [3,4,5]]
__(data).get(0).get('key1dd', {}).__



@dispatch(dict, str)
def test(data,fields,pattern=None):
    print(data, fields)
    print('1111pattern', pattern)

@dispatch(dict, str, str)
def test(data,fields,pattern=None):
    print(data, fields)
    print('pattern', pattern)


test({'22': 'ff'}, 'ggg')
test({'22': 'ff'}, 'ggg', None)

def f(a, b, c, d=None, e='aa'):
    sig = inspect.signature(f)
    parameters = sig.parameters
    bound_args = sig.bind(a, b, c, d=d, e=e)
    bound_args.apply_defaults()

    # Identify parameters provided explicitly by the user
    user_provided = {name: value for name, value in bound_args.arguments.items()
                     if name not in parameters or parameters[name].default != value}

    # Do stuff
    print("User provided:", user_provided)

# Example usage
f(1, 2, 3)
f(1, 2, 3, d=4)
f(1, 2, 3, e='bb')


sample_data = [
    [{'thekey': 'VALUE1', 'otherkey': 'data'}, {'thekey': 'VALUE2'}],
    [{'anotherkey': 123}, {'thekey': 'SOMETHING'}, {'notthekey': 'nope'}],
    [{'thekey': 'HELLO WORLD'}]
]



@dispatch(Iterable,FunctionType)
def g(x, func):
    @dispatch(Mapping)
    def ee(x):
        return [y for y in x.items() if func(y)]
    def ee(x):
        return [y for y in x if func(y)]



@dispatch(Iterable,FunctionType)
def g(x, func):
    if isinstance(x, Mapping):
        x =  x.items()
    return [y for y in x if func(y)]
 

g([1,2,3,2,2], lambda x:x ==2)
g({'key1': 'aaa', 'key2': 'gggg', 'key3': 'dddd'}, lambda x:x[0] == 'key1')
g({'key1': 'aaa', 'key2': 'gggg', 'key3': 'dddd'}, lambda x:x[1] == 'dddd')


@dispatch(Iterable,FunctionType)
def g(x):
    return x==2

isinstance({}, Mapping)

g([1,2,3,2,2], lambda x: x==2)
g({'key1': 'aaa', 'key2': 'gggg', 'key3': 'dddd'}, lambda x: x == 'key1')

f([1,2,3,5])

@dispatch(Iterable)
def flatten(L):
    return sum([flatten(x) for x in L], [])


@dispatch(int, int)
def gg(a, b=10):
    print(a, b)

gg.funcs

@dispatch(int, int)
@dispatch(int)
def foo(a, b=10):
    print(a, b)

foo(2,b=40)

[[{key: value.lower() if key == 'thekey' else value for key, value in dct.items()} for dct in nested_array ] for nested_array in sample_data]

[[{key: value.lower() if key == 'thekey' else value for key, value in dct.items()} for dct in nested_array if dct.get('otherkey') == 'data'] for nested_array in arrays]

__(sample_data).get('thekey')

@dispatch(dict, str)
def test(data, fields, pattern=None):
    print(data, fields)
    print('pattern', pattern)

@dispatch(dict, str, str)
def test(data, fields, pattern):
    print(data, fields)
    print('pattern333', pattern)

# Now you can call the function with three arguments:
test({'22': 'ff'}, 'ggg')

test({'22': 'ff'}, 'ggg', 'aaa')

_get(data,1,safe={}).safe

(safe=None)

isinstance()
isinstance(__(data).get(0).get(6).data, type(Nothing))

a=Nothing

Nothing.value_or(2)

__(Some(data[88]))


__(data).get(0).get(0)

.get(.__, 0)


top_students = __(students
).filter(
    lambda student: all(grade >= 60 for grade in student['grades'].values())
).val.apply(
    lambda student: {
        'name': student['name'],
        'average': sum(student['grades'].values()) / len(student['grades'])
    }
).sort(
    key=lambda x: x['average'],
    reverse=True
).get(
    slice(0, 3)
).__


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

result = __(matrix).flatten()


@dispatch((int,float), int)
@dispatch(int, int)
def add(x, y):
    return x + y



@dispatch(object, object)
def add(x, y):
    return "%s + %s" % (x, y)

add(1, 2)
add(1.0,5)

@(dict,FunctionType)






__({'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}).get(lambda t: t[1] == 'aaa')

data={'key1': 'aaa', 'key2': 'bbb', 'key3': 'gggg'}
_get(data,lambda t: t[1] == 'aaa')
_get(data,'key1')


_get(data, 3)


add(1, 'hello')

@dispatch(int, int)
@dispatch(int)
def foo(a, b=10):
    print(a,b)


foo(2, 6)

from functools import singledispatch

@singledispatch
def process(data, field):
    raise NotImplementedError("Unsupported type")

@process.register(int, int)
def _(data):
    return f"Integer: {data}"

@process.register(int, str)
def _(data, ff):
    return f"Integer: {data} ;;; {ff}"

# @process.register(str)
# def _(data):
#     return f"String: {data}"

# @process.register(list)
# def _(data):
#     return f"List with length {len(data)}"

# Usage
print(process(123))       # Outputs: Integer: 123
print(process(123, 'aaaaaa'))       # Outputs: Integer: 123
print(process("hello"))   # Outputs: String: hello
print(process([1, 2, 3])) # Outputs: List with length 3





__(hierarchy).inplace.get(['properties', 'references'])

def safe_jmes_search(query, data) -> Maybe:
    result = jmespath.search(query, data)
    if result is not None:
        return Some(result)
    return Nothing


def jmesOrNothing(query, data) -> Maybe:
    result = jmespath.search(query, data)
    if result is not None:
        return Some(result)
    return Nothing
    
class BooleanTest:
    def __init__(self, func):
        self.func = func

    def trueOrRaise(self, message):
        if not self.func():
            raise ValueError(message)
        return True

# @dispatch(Iterable,FunctionType)
# def get(data, func):
#     print('PPPPP')
#     if isinstance(data, Mapping):
#         print('aaa')
#         data = data.items()
#     return __([y for y in data if func(y)])






list(chain.from_iterable(map(lambda key: map(grapql_to_d3hierarchy_format, sub_level[key]), keys)))


map(lambda key: map(grapql_to_d3hierarchy_format, sub_level[key]), keys)


[grapql_to_d3hierarchy_format(child) for key in keys for child in sub_level[key]]

keys=['a', 'b', 'c']
levels=[{'a': 2, 'b': 3, 'c': 4},{'a': 5, 'b': 6, 'c': 7}]

# func=lambda x: levels[x]+2

func=lambda x: x+2

map(func, levels[key])

[func(i) for i in keys]

__(keys).map(lambda k: func(levels[k])).__

__(keys).map(lambda k: 
             __(levels[k]).map(lambda x: func)).__

__(levels).map(lambda level: 
             __(keys).map(lambda key: func(level[key])).__).__

__(levels).map(
    __(keys).map(lambda key: func(level[key])).__).__


list(chain.from_iterable(map(lambda key: map(func, levels[key]), keys)))
list(chain.from_iterable(map(lambda key: func(levels[key]), keys)))

list(map(lambda key: func(levels[key]), keys))

__(keys).map(lambda k: levels[k])

__(keys).map(lambda k: k)

__(keys).map(lambda k: levels[k]+2)

def g(x):
    return x+4

__(keys).map(lambda key:
             __(sub_level[key]).map(g))

__(keys).map(lambda key:
             __(sub_level[key]).map(grapql_to_d3hierarchy_format))


data=[[1,2,3], [4,5,6], [7,8,9]]
__(data).map(lambda key:key+1)

__(data).map(lambda key:key+1)


