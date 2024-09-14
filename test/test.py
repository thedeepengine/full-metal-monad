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

if __name__ == '__main__':
    unittest.main()
