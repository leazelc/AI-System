import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "memory"))

from memory import keyword_index
from memory_retrieval_orchestrator import (
    _retrieve_once,
    plan_queries,
    readonly_keyword_search,
)


class TestRetrievalPlanning(unittest.TestCase):
    def test_planning_keeps_original_query(self):
        def fake_model(system, prompt):
            return {"queries": ["预算是多少", "电脑配置偏好"]}

        queries = plan_queries(
            "我的电脑预算是什么？",
            model_json_fn=fake_model,
        )

        self.assertEqual(queries[0], "我的电脑预算是什么？")
        self.assertEqual(len(queries), 3)

    def test_planning_falls_back_when_model_fails(self):
        def failing_model(system, prompt):
            raise RuntimeError("model unavailable")

        queries = plan_queries(
            "我之前的电脑预算是多少？",
            model_json_fn=failing_model,
        )

        self.assertEqual(queries, ["我之前的电脑预算是多少？"])

    def test_planning_removes_duplicate_queries(self):
        def fake_model(system, prompt):
            return {
                "queries": [
                    "电脑预算",
                    "电脑预算",
                    "",
                    None,
                    "SSD 偏好",
                ]
            }

        queries = plan_queries(
            "我的电脑预算是什么？",
            model_json_fn=fake_model,
        )

        self.assertEqual(
            queries,
            ["我的电脑预算是什么？", "电脑预算", "SSD 偏好"],
        )


class TestReadonlyKeywordSearch(unittest.TestCase):
    def test_missing_fts_index_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            db_path = Path(temp) / "test.sqlite"

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        status TEXT NOT NULL
                    )
                """)

            results = readonly_keyword_search(
                "电脑预算",
                db_path=db_path,
            )

            self.assertEqual(results, [])

            with sqlite3.connect(db_path) as conn:
                names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master"
                    )
                }

            self.assertNotIn("memory_keyword_fts", names)

    def test_existing_fts_index_returns_active_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "test.sqlite"
            memory_path = root / "budget.md"
            memory_path.write_text(
                "我的电脑预算上限是两万元人民币。",
                encoding="utf-8",
            )

            with sqlite3.connect(db_path) as conn:
                conn.executescript("""
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        status TEXT NOT NULL
                    );

                    CREATE VIRTUAL TABLE memory_keyword_fts
                    USING fts5(id UNINDEXED, content, tokenize='unicode61');
                """)

                conn.execute(
                    "INSERT INTO memories VALUES (?, ?, ?)",
                    ("budget", str(memory_path), "active"),
                )
                conn.execute(
                    "INSERT INTO memory_keyword_fts VALUES (?, ?)",
                    (
                        "budget",
                        keyword_index.make_terms(
                            memory_path.read_text(encoding="utf-8")
                        ),
                    ),
                )

            results = readonly_keyword_search(
                "电脑预算",
                db_path=db_path,
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["id"], "budget")


class TestHybridCandidateFusion(unittest.TestCase):
    def test_rrf_merges_semantic_and_keyword_rankings(self):
        semantic_results = [
            {
                "id": "a",
                "path": "/tmp/a.md",
                "score": 0.82,
                "metadata": {"status": "active"},
                "content": "A 记忆正文",
            },
            {
                "id": "b",
                "path": "/tmp/b.md",
                "score": 0.76,
                "metadata": {"status": "active"},
                "content": "B 记忆正文",
            },
        ]

        keyword_results = [
            {
                "id": "b",
                "path": Path("/tmp/b.md"),
                "score": 1.5,
            },
            {
                "id": "a",
                "path": Path("/tmp/a.md"),
                "score": 1.2,
            },
        ]

        def fake_semantic(query, limit=10, min_score=None):
            return semantic_results

        def fake_keyword(query, limit=10):
            return keyword_results

        results = _retrieve_once(
            ["预算"],
            semantic_search_fn=fake_semantic,
            keyword_search_fn=fake_keyword,
            candidate_limit=10,
        )

        by_id = {item["id"]: item for item in results}

        self.assertEqual(set(by_id), {"a", "b"})
        self.assertGreater(by_id["a"]["score"], by_id["b"]["score"])
        self.assertGreater(by_id["a"]["rrf_score"], 0)
        self.assertGreater(by_id["b"]["rrf_score"], 0)

    def test_keyword_only_candidate_is_not_silently_lost(self):
        semantic_results = [
            {
                "id": "semantic",
                "path": "/tmp/semantic.md",
                "score": 0.81,
                "metadata": {"status": "active"},
                "content": "语义检索命中的记忆",
            }
        ]

        keyword_results = [
            {
                "id": "keyword-only",
                "path": Path("/tmp/keyword-only.md"),
                "score": 2.0,
            }
        ]

        def fake_semantic(query, limit=10, min_score=None):
            return semantic_results

        def fake_keyword(query, limit=10):
            return keyword_results

        # 当前设计应保留关键词独有候选，并读取其真实正文与元数据。
        # 这项测试用于确认该行为，若失败则说明协调器需要修复。
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "keyword-only.md"
            path.write_text(
                "---\nstatus: active\nid: keyword-only\n---\n"
                "用户明确记录的电脑预算与设备偏好。",
                encoding="utf-8",
            )

            keyword_results[0]["path"] = path

            results = _retrieve_once(
                ["电脑预算"],
                semantic_search_fn=fake_semantic,
                keyword_search_fn=fake_keyword,
                keyword_candidate_score_fn=lambda query, items, semantic_ids: {
                    "keyword-only": {
                        "id": "keyword-only",
                        "path": path,
                        "score": 0.63,
                        "metadata": {"status": "active"},
                        "content": "用户明确记录的电脑预算与设备偏好。",
                    }
                },
                candidate_limit=10,
            )

        by_id = {item["id"]: item for item in results}
        self.assertIn("keyword-only", by_id)
        self.assertEqual(
            by_id["keyword-only"]["content"],
            "用户明确记录的电脑预算与设备偏好。",
        )
        self.assertEqual(
            by_id["keyword-only"]["metadata"].get("status"),
            "active",
        )


class TestRetrievalEndToEnd(unittest.TestCase):
    def test_empty_question_does_not_call_model_or_retrieval(self):
        from memory_retrieval_orchestrator import retrieve

        def unexpected(*args, **kwargs):
            raise AssertionError("空问题不应触发模型或检索")

        result = retrieve(
            "   ",
            model_json_fn=unexpected,
            semantic_search_fn=unexpected,
            keyword_search_fn=unexpected,
        )

        self.assertEqual(result["memories"], [])
        self.assertEqual(result["queries"], [])
        self.assertFalse(result["diagnostics"]["analysis_used_ai"])

    def test_followup_retrieval_merges_new_and_duplicate_memories(self):
        from memory_retrieval_orchestrator import retrieve

        model_responses = [
            {"queries": ["预算相关记忆"]},
            {
                "relevant_ids": ["a", "not-a-real-id"],
                "missing_aspects": ["还缺少最终确定的预算"],
                "followup_queries": ["最终预算"],
            },
            {
                "relevant_ids": ["b"],
                "missing_aspects": [],
                "followup_queries": [],
            },
        ]

        def fake_model(system, prompt):
            if not model_responses:
                raise AssertionError("不应发生额外模型调用")
            return model_responses.pop(0)

        def fake_semantic(query, limit=10, min_score=None):
            items = {
                "a": {
                    "id": "a",
                    "path": "/tmp/a.md",
                    "score": 0.80,
                    "metadata": {"status": "active"},
                    "content": "记忆 A",
                }
            }

            if query == "最终预算":
                items["b"] = {
                    "id": "b",
                    "path": "/tmp/b.md",
                    "score": 0.72,
                    "metadata": {"status": "active"},
                    "content": "记忆 B",
                }

            return list(items.values())

        def no_keyword_results(query, limit=10):
            return []

        result = retrieve(
            "我的预算是多少？",
            model_json_fn=fake_model,
            semantic_search_fn=fake_semantic,
            keyword_search_fn=no_keyword_results,
        )

        ids = [item["id"] for item in result["memories"]]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {"a", "b"})
        self.assertEqual(result["queries"][0], "我的预算是多少？")
        self.assertEqual(
            result["diagnostics"]["followup_queries_used"],
            ["最终预算"],
        )
        self.assertNotIn("not-a-real-id", result["analysis"]["relevant_ids"])
        self.assertEqual(result["analysis"]["relevant_ids"], ["b"])
        self.assertEqual(result["diagnostics"]["analysis_status"], "success")
        self.assertEqual(model_responses, [])

    def test_ai_relevance_changes_final_ranking(self):
        from memory_retrieval_orchestrator import retrieve

        responses = [
            {"queries": []},
            {
                "relevant_ids": ["lower-ranked"],
                "missing_aspects": [],
                "followup_queries": [],
            },
        ]

        def fake_model(system, prompt):
            return responses.pop(0)

        def fake_semantic(query, limit=10, min_score=None):
            return [
                {
                    "id": "higher-rrf",
                    "path": "/tmp/a.md",
                    "score": 0.9,
                    "metadata": {},
                    "content": "候选 A",
                },
                {
                    "id": "lower-ranked",
                    "path": "/tmp/b.md",
                    "score": 0.7,
                    "metadata": {},
                    "content": "候选 B",
                },
            ]

        result = retrieve(
            "测试问题",
            model_json_fn=fake_model,
            semantic_search_fn=fake_semantic,
            keyword_search_fn=lambda query, limit=10: [],
        )

        self.assertEqual(
            [item["id"] for item in result["memories"]],
            ["lower-ranked", "higher-rrf"],
        )
        self.assertTrue(result["diagnostics"]["analysis_used_ai"])
        self.assertEqual(result["diagnostics"]["analysis_status"], "success")

    def test_ai_relevance_order_is_preserved_within_selected_candidates(self):
        from memory_retrieval_orchestrator import retrieve

        responses = [
            {"queries": []},
            {
                "relevant_ids": ["third-by-score", "second-by-score"],
                "missing_aspects": [],
                "followup_queries": [],
            },
        ]

        def fake_model(system, prompt):
            return responses.pop(0)

        def fake_semantic(query, limit=10, min_score=None):
            return [
                {
                    "id": "first-by-score",
                    "path": "/tmp/a.md",
                    "score": 0.99,
                    "metadata": {},
                    "content": "传统排名第一",
                },
                {
                    "id": "second-by-score",
                    "path": "/tmp/b.md",
                    "score": 0.90,
                    "metadata": {},
                    "content": "传统排名第二",
                },
                {
                    "id": "third-by-score",
                    "path": "/tmp/c.md",
                    "score": 0.80,
                    "metadata": {},
                    "content": "AI 判断最相关",
                },
            ]

        result = retrieve(
            "测试 AI 相关性排序",
            model_json_fn=fake_model,
            semantic_search_fn=fake_semantic,
            keyword_search_fn=lambda query, limit=10: [],
        )

        ids = [item["id"] for item in result["memories"]]
        self.assertEqual(
            ids,
            ["third-by-score", "second-by-score", "first-by-score"],
        )
        self.assertEqual(
            result["analysis"]["relevant_ids"],
            ["third-by-score", "second-by-score"],
        )


    def test_analysis_rejects_non_object_model_response(self):
        from memory_retrieval_orchestrator import analyze_candidates

        candidates = [{
            "id": "a",
            "score": 0.8,
            "content": "真实候选记忆",
        }]

        result = analyze_candidates(
            "测试问题",
            candidates,
            model_json_fn=lambda system, prompt: ["不是 JSON 对象"],
        )

        self.assertEqual(result["relevant_ids"], [])
        self.assertEqual(result["missing_aspects"], [])
        self.assertEqual(result["followup_queries"], [])

    def test_analysis_model_exception_does_not_mark_all_candidates_relevant(self):
        from memory_retrieval_orchestrator import analyze_candidates

        candidates = [
            {"id": "a", "score": 0.8, "content": "记忆 A"},
            {"id": "b", "score": 0.7, "content": "记忆 B"},
        ]
        status = {"used_ai": False}

        def failed_model(system, prompt):
            raise RuntimeError("local model unavailable")

        result = analyze_candidates(
            "测试问题", candidates,
            model_json_fn=failed_model,
            status=status,
        )

        self.assertEqual(result["relevant_ids"], [])
        self.assertFalse(status["used_ai"])
        self.assertEqual(status["analysis_status"], "model_error")

    def test_analysis_ignores_unknown_ids_and_invalid_fields(self):
        from memory_retrieval_orchestrator import analyze_candidates

        candidates = [{
            "id": "real-id",
            "score": 0.8,
            "content": "真实候选记忆",
        }]

        result = analyze_candidates(
            "测试问题",
            candidates,
            model_json_fn=lambda system, prompt: {
                "relevant_ids": ["unknown-id", "real-id", 123],
                "missing_aspects": [" 缺少日期 ", None, 123],
                "followup_queries": ["补充日期", "补充日期"],
            },
        )

        self.assertEqual(result["relevant_ids"], ["real-id"])
        self.assertEqual(result["missing_aspects"], ["缺少日期"])
        self.assertEqual(result["followup_queries"], ["补充日期"])

    def test_followup_limit_zero_disables_second_round(self):
        from memory_retrieval_orchestrator import retrieve

        model_responses = [
            {"queries": []},
            {
                "relevant_ids": ["a"],
                "missing_aspects": ["还缺日期"],
                "followup_queries": ["查询日期"],
            },
        ]

        def fake_model(system, prompt):
            return model_responses.pop(0)

        def fake_semantic(query, limit=10, min_score=None):
            return [{
                "id": "a",
                "path": "/tmp/a.md",
                "score": 0.8,
                "metadata": {"status": "active"},
                "content": "记忆 A",
            }]

        result = retrieve(
            "测试问题",
            model_json_fn=fake_model,
            semantic_search_fn=fake_semantic,
            keyword_search_fn=lambda query, limit=10: [],
            followup_limit=0,
        )

        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["diagnostics"]["followup_queries_used"], [])
        self.assertEqual(len(model_responses), 0)




class TestAIRetrievalDiagnostics(unittest.TestCase):
    @staticmethod
    def semantic_results(query, limit=10, min_score=None):
        return [{
            "id": "memory-a",
            "path": "/tmp/memory-a.md",
            "score": 0.82,
            "metadata": {"status": "active"},
            "content": "用户之前讨论过电脑预算。",
        }]

    @staticmethod
    def no_keyword_results(query, limit=10):
        return []

    def test_failed_planning_is_not_reported_as_ai_success(self):
        def failing_model(system, prompt):
            raise RuntimeError("模型不可用")

        from memory_retrieval_orchestrator import retrieve

        result = retrieve(
            "之前的电脑预算是多少？",
            model_json_fn=failing_model,
            semantic_search_fn=self.semantic_results,
            keyword_search_fn=self.no_keyword_results,
        )

        self.assertFalse(result["diagnostics"]["planning_used_ai"])
        self.assertFalse(result["diagnostics"]["analysis_used_ai"])
        self.assertEqual(result["queries"], ["之前的电脑预算是多少？"])

    def test_non_object_analysis_response_is_not_reported_as_success(self):
        responses = [
            {"queries": ["之前的电脑预算"]},
            ["这不是有效的分析对象"],
        ]

        def fake_model(system, prompt):
            return responses.pop(0)

        from memory_retrieval_orchestrator import retrieve

        result = retrieve(
            "之前的电脑预算是多少？",
            model_json_fn=fake_model,
            semantic_search_fn=self.semantic_results,
            keyword_search_fn=self.no_keyword_results,
            followup_limit=0,
        )

        self.assertTrue(result["diagnostics"]["planning_used_ai"])
        self.assertFalse(result["diagnostics"]["analysis_used_ai"])

    def test_query_limit_one_skips_planning_but_keeps_ai_analysis(self):
        from memory_retrieval_orchestrator import retrieve

        model_calls = []

        def fake_model(system, prompt):
            model_calls.append((system, prompt))

            # 查询规划不应调用模型；唯一一次调用应是候选分析。
            self.assertIn("Retrieved candidates (JSON):", prompt)
            return {
                "relevant_ids": ["memory-a"],
                "missing_aspects": [],
                "followup_queries": [],
            }

        result = retrieve(
            "之前的电脑预算是多少？",
            model_json_fn=fake_model,
            semantic_search_fn=self.semantic_results,
            keyword_search_fn=self.no_keyword_results,
            query_limit=1,
            followup_limit=0,
        )

        self.assertEqual(
            result["queries"],
            ["之前的电脑预算是多少？"],
        )
        self.assertEqual(len(model_calls), 1)
        self.assertFalse(result["diagnostics"]["planning_used_ai"])
        self.assertTrue(result["diagnostics"]["analysis_used_ai"])
        self.assertEqual(
            result["diagnostics"]["analysis_status"],
            "success",
        )
        self.assertEqual(
            [item["id"] for item in result["memories"]],
            ["memory-a"],
        )


    def test_valid_planning_and_analysis_are_reported_as_success(self):
        responses = [
            {"queries": ["之前的电脑预算"]},
            {
                "relevant_ids": ["memory-a"],
                "missing_aspects": [],
                "followup_queries": [],
            },
        ]

        def fake_model(system, prompt):
            return responses.pop(0)

        from memory_retrieval_orchestrator import retrieve

        result = retrieve(
            "之前的电脑预算是多少？",
            model_json_fn=fake_model,
            semantic_search_fn=self.semantic_results,
            keyword_search_fn=self.no_keyword_results,
            followup_limit=0,
        )

        self.assertTrue(result["diagnostics"]["planning_used_ai"])
        self.assertTrue(result["diagnostics"]["analysis_used_ai"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
