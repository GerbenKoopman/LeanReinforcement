import unittest
from ReProver.common_cy import Pos


class TestPosCy(unittest.TestCase):
    def test_pos_initialization(self):
        pos = Pos(1, 2)
        self.assertEqual(pos.line, 1)
        self.assertEqual(pos.column, 2)

    def test_pos_equality(self):
        pos1 = Pos(1, 2)
        pos2 = Pos(1, 2)
        pos3 = Pos(2, 3)
        self.assertEqual(pos1, pos2)
        self.assertNotEqual(pos1, pos3)
        self.assertNotEqual(pos1, "not a pos")

    def test_pos_hash(self):
        pos1 = Pos(1, 2)
        pos2 = Pos(1, 2)
        self.assertEqual(hash(pos1), hash(pos2))

        s = {pos1}
        self.assertIn(pos2, s)

    def test_pos_repr(self):
        pos = Pos(1, 2)
        self.assertEqual(repr(pos), "Pos(line=1, column=2)")
