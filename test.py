import function
import unittest
#a test class can established inherited from unittest.TestCase

class TestAddition(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(addition(1,2),3)
    def test_divition(self):
        with self.assertRaises(ValueError):
            function.divide(10,2)

if __name__ == '__main__':
    unittest.main()
#use test to verify gradient descents