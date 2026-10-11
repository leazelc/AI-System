import unittest

from memory_proxy import is_personal_memory_question


class TestPersonalMemoryQuestionClassifier(unittest.TestCase):

    def test_recognizes_explicit_chinese_personal_memory_questions(self):
        questions = [
            "我之前设定的私人保险箱编号是多少？",
            "你还记得我的专业吗？",
            "我们上次讨论了什么？",
            "我以前告诉过你什么？",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(
                    is_personal_memory_question(question),
                    f"应该识别为个人记忆问题：{question}",
                )

    def test_recognizes_english_personal_memory_questions(self):
        questions = [
            "What was my previous plan?",
            "What is my major?",
            "What do you remember about me?",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(
                    is_personal_memory_question(question),
                    f"应该识别为个人记忆问题：{question}",
                )

    def test_does_not_classify_ordinary_questions_as_personal_memory(self):
        questions = [
            "1+1 等于多少？",
            "什么是社会学？",
            "What is sociology?",
            "How does photosynthesis work?",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(
                    is_personal_memory_question(question),
                    f"普通问题不应该被识别为个人记忆问题：{question}",
                )


if __name__ == "__main__":
    unittest.main()
