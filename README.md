# Full Metal Monad


## Installation

- From github:

```
pip install git+https://github.com/thedeepengine/full-metal-monad.git
```


When getting some data out of a list or dict, I dont really that the syntax should be this if its an array, or that if its a dict, what really matters is that the element I am trying to access exist and accessing it.

Thats weird because it seems liek the focus is on using the correct syntax. this is what you have to focus on.

With full metal monad, you dont focus on the specific syntax of the data structure, if its a list or a dict, there is a single interface for example .get. the focus is on making sure your code is safer as its wrapped in monads, that you can chain.

Also you can vectorize functions.
For example

[[{key: value.lower() if key == 'thekey' else value for key, value in dct.items()} for dct in nested_array ] for nested_array in arrays]


## Main Axis this library care about

the goal is to commoditize or streamline 

- data operations on native python data structure
- vectorization 
- exception handling


Couple of odd things:

```{python}
# Example array which may contain dictionaries
array = [{}]

try:
    # Attempt to access the 'key' in the first dictionary if it exists
    value = array[0].get('key')
    if value is not None:
        return value
    else:
        return {}
except IndexError:
    print("Array is empty")
```

Here:
- empty array check is done through try-except
- missing key is checked through if-else

with full metal monad

```
array = [{}]

__(array).get(0).get('key').get_or_raise()

__(array).get(0).get('key').__

```


Eventually a more elegant way to handle that woudl be:

```
if array:  # Check if the array is not empty
    first_dict = array[0]
    if 'key' in first_dict:  # Check if the key exists in the dictionary
        value = first_dict['key']
        print("Value found:", value)
    else:
        print("Key not found")
else:
    print("Array is empty")
```
