import unittest
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer


class TestVanillaBuffer(unittest.TestCase):

    def setUp(self):
        self.trans = UniformBuffer.Transition(
            np.ones(24, dtype="float"),
            1,
            1.0,
            -np.ones(24, dtype="float"),
            True,
        )

    def test_sample(self):
        capacity = 1000
        buffer = UniformBuffer(capacity, (24,), "float")
        for i in range(10):
            buffer.push(self.trans)
        sample, = buffer.sample(5)
        self.assertIsInstance(sample, UniformBuffer.Transition)

        self.assertEqual(buffer.sample(11), None)

        for item, info in zip(sample, buffer.transition_info):
            self.assertEqual(item.shape, (5, *info["shape"]))
            self.assertEqual(item.dtype, info["dtype"])

        for i in range(100):
            self.assertEqual(buffer.sample(6)[0].state.sum(), 24.0 * 6)

    def test_write_index_and_size(self):
        capacity = 1000
        buffer = UniformBuffer(capacity, (24,), "float")
        for i in range(capacity * 2):
            self.assertEqual(i % capacity, buffer.write_index)
            buffer.push(self.trans)
            self.assertEqual(min(i+1, capacity), buffer.size)
            self.assertEqual(buffer.buffer.state.sum().item() // 24,
                             min(i+1, capacity))


class TestPrioritizedBuffer(unittest.TestCase):

    def setUp(self):
        self.trans_pos = UniformBuffer.Transition(
            np.ones(24, dtype="float"),
            1,
            1.0,
            -np.ones(24, dtype="float"),
            True,
        )

        self.trans_neg = UniformBuffer.Transition(
            -np.ones(24, dtype="float"),
            1,
            1.0,
            np.ones(24, dtype="float"),
            True,
        )

    def test_overwriting(self):
        """ Test if Priority Buffer is overwriting """
        buffer = PriorityBuffer(500, (24,), "float", alpha=0.5, epsilon=4)
        for _ in range(500):
            buffer.push(self.trans_pos)
        for _ in range(500):
            buffer.push(self.trans_neg)

        self.assertEqual(buffer.abs_td_errors.sum(), 2000)
        batch, indices, weights = buffer.sample(500, 0.5)
        self.assertEqual(batch.state.sum() / 24, -500)

    def test_sample_indexes(self):
        """ Test if indices are overflowing """
        buffer = PriorityBuffer(500, (24,), "float", alpha=0.5, epsilon=4)
        for _ in range(100):
            buffer.push(self.trans_pos)
        for _ in range(10):
            batch, indices, weights = buffer.sample(10, 0.5)
            self.assertEqual(np.all(
                indices < 100
            ), 1)

    def test_update(self):
        """ Test update_priority method and max_priority """
        buffer = PriorityBuffer(7, (24,), "float", alpha=0.5, epsilon=4)
        for _ in range(4):
            buffer.push(self.trans_pos)
        buffer.update_priority(
            [0, 1], [12, 12]
        )
        self.assertEqual(buffer.abs_td_errors.sum(), (12 + 4) + (12 + 4) + 4 + 4)
        buffer.push(self.trans_pos)
        self.assertEqual(buffer.abs_td_errors.sum(), (12 + 4) + (12 + 4) + 4 + 4 + 16)

    def test_weights(self):
        """ Test weights returned by sample method """
        buffer = PriorityBuffer(9, (24,), "float", alpha=0.5, epsilon=4)
        for i in range(4):
            buffer.push(self.trans_pos)

        batch, indices, weights = buffer.sample(4, 0.5)
        self.assertTrue(np.allclose(weights, np.array([1, 1, 1, 1])))


if __name__ == '__main__':
    unittest.main()
