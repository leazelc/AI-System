import unittest

from memory.memory_validation import validate_decision


def valid_new_decision():
    return {
        "action": "NEW",
        "target_id": None,
        "type": "preferences",
        "importance": 0.8,
        "confidence": 0.95,
        "memory": "个人 AI 数据优先保存在本地。",
    }


class TestMemoryValidation(unittest.TestCase):
    def test_valid_new_decision(self):
        self.assertEqual(
            validate_decision(valid_new_decision()),
            (True, None),
        )

    def test_invalid_action(self):
        decision = valid_new_decision()
        decision["action"] = "DELETE"
        self.assertFalse(validate_decision(decision)[0])

    def test_empty_memory(self):
        decision = valid_new_decision()
        decision["memory"] = "   "
        self.assertFalse(validate_decision(decision)[0])

    def test_invalid_type(self):
        decision = valid_new_decision()

        decision["type"] = "random"
        self.assertFalse(validate_decision(decision)[0])

    def test_plans_type_is_valid(self):
        decision = valid_new_decision()
        decision["type"] = "plans"
        self.assertEqual(validate_decision(decision), (True, None))

    def test_technical_config_type_is_valid(self):
        decision = valid_new_decision()
        decision["type"] = "technical_config"
        self.assertEqual(validate_decision(decision), (True, None))

    def test_confidence_out_of_range(self):
        decision = valid_new_decision()
        decision["confidence"] = 1.5
        self.assertFalse(validate_decision(decision)[0])

    def test_nan_is_rejected(self):
        decision = valid_new_decision()
        decision["importance"] = float("nan")
        self.assertFalse(validate_decision(decision)[0])

    def test_update_requires_target_id(self):
        decision = valid_new_decision()
        decision["action"] = "UPDATE"
        self.assertFalse(validate_decision(decision)[0])

    def test_update_rejects_unknown_target(self):
        decision = valid_new_decision()
        decision["action"] = "UPDATE"
        decision["target_id"] = "mem_unknown"
        self.assertFalse(
            validate_decision(decision, valid_ids={"mem_existing"})[0]
        )

    def test_ignore_requires_empty_memory(self):
        decision = valid_new_decision()
        decision["action"] = "IGNORE"
        decision["memory"] = None
        self.assertTrue(validate_decision(decision)[0])

        decision["memory"] = "不应该保存的内容"
        self.assertFalse(validate_decision(decision)[0])


    def test_boolean_importance_is_rejected(self):
        decision = valid_new_decision()
        decision["importance"] = True
        self.assertFalse(validate_decision(decision)[0])

    def test_negative_confidence_is_rejected(self):
        decision = valid_new_decision()
        decision["confidence"] = -0.1
        self.assertFalse(validate_decision(decision)[0])

    def test_new_rejects_target_id(self):
        decision = valid_new_decision()
        decision["target_id"] = "mem_existing"
        self.assertFalse(validate_decision(decision)[0])

    def test_merge_requires_target_id(self):
        decision = valid_new_decision()
        decision["action"] = "MERGE"
        self.assertFalse(validate_decision(decision)[0])


if __name__ == "__main__":
    unittest.main()
