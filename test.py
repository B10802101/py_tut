from function import addition
import unittest
#a test class can established inherited from unittest.TestCase

class TestAddition(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(addition(1,2),3)

if __name__ == '__main__':
    unittest.main()
#use test to verify gradient descent