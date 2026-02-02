import sys

sys.path.append("..")
import ast
import unittest

from src import ReverseIfElser
from tests.base_tranform_test import baseCodeTransformTest



    # TESTS
    def test_no_if_else(self):
        source_code = """
def greet(language):
    greetings = {
        "en": "Hello",
        "es": "Hola",
        "fr": "Bonjour"
    }
    return greetings.get(language, "Hello")

numbers = [1, 2, 3, 4, 5]
even_numbers = [num for num in numbers if num % 2 == 0]

print(even_numbers)
print(greet("es"))
        """

        expected_code = """
def greet(language):
    greetings = {
        "en": "Hello",
        "es": "Hola",
        "fr": "Bonjour"
    }
    return greetings.get(language, "Hello")
numbers = [1, 2, 3, 4, 5]
even_numbers = [num for num in numbers if num % 2 == 0]
print(even_numbers)
print(greet("es"))
        """

        self.assert_code_equal(
            self.get_transformed_code(source_code, ReverseIfElser), expected_code
        )

    def test_reverse_if_else_with_single_if_branch(self):
        source_code = """
name = "Alice"
is_raining = True

if name == "Alice":
    print("Hello, Alice!")

if is_raining:
    print("Remember to take an umbrella!")
"""
        expected_code = """
name = "Alice"
is_raining = True

if not (name == "Alice"):
    ...
else:
    print("Hello, Alice!")

if not is_raining:
    ...
else:
    print("Remember to take an umbrella!")
"""
        self.assert_code_equal(
            self.get_transformed_code(source_code, ReverseIfElser), expected_code
        )

    def test_reverse_if_else_with_single_if_Ands_branch(self):
        source_code = """
true_flag, false_flag = True, False

if true_flag and not false_flag:
    print("true flags")
"""
        expected_code = """
true_flag, false_flag = True, False

if not (true_flag and not false_flag):
    ...
else:
    print("true flags")
"""
        self.assert_code_equal(
            self.get_transformed_code(source_code, ReverseIfElser), expected_code
        )

    def test_reverse_if_else_with_else_branch(self):
        source_code = """
if x > 5:
    print("Greater")
else:
    print("Smaller or Equal")
"""
        expected_code = """
if not (x > 5):
    print("Smaller or Equal")
else:
    print("Greater")
"""
        self.assert_code_equal(
            self.get_transformed_code(source_code, ReverseIfElser), expected_code
        )

    def test_reverse_if_else_with_nested_if_else_branch(self):
        source_code = """
x = 2
if x < 3:
    print('smaller than 3')
elif x < 4:
    print('bigger than or equal 3 AND smaller than 4')
elif x < 5:
    print('bigger than or equal to 4 AND smaller than 5')
else:
    print('bigger than or equal to 5')
"""
        expected_code = """
x = 2
if not x < 3:
    if not x < 4:
        if not x < 5:
            print('bigger than or equal to 5')
        else:
            print('bigger than or equal to 4 AND smaller than 5')
    else:
        print('bigger than or equal 3 AND smaller than 4')
else:
    print('smaller than 3')
"""
        self.assert_code_equal(
            self.get_transformed_code(source_code, ReverseIfElser), expected_code
        )


if __name__ == "main":
    unittest.main()
