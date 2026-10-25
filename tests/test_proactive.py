from __future__ import annotations

import unittest

from tests.helpers import make_engine


def add_fact(engine, text, *, kind="fact", pool="cycle", importance=0.9):
    engine.db.execute(
        "INSERT INTO facts(user_key,text,kind,importance,created_at,pool)"
        " VALUES(?,?,?,?,?,?)",
        ("u1", text, kind, importance, engine.clock(), pool),
    )


class ProactiveContentTests(unittest.TestCase):
    def test_identity_and_fragment_facts_are_not_used(self):
        engine, _ = make_engine()
        add_fact(engine, "你叫“蓝心心”", kind="assistant_name")
        add_fact(engine, "提到“有啊”")
        add_fact(engine, "爱你", kind="habit")
        self.assertEqual("", engine.proactive._top_fact("u1"))
        text, kind = engine.proactive.compose("u1")
        self.assertEqual("presence", kind)
        self.assertTrue(text)

    def test_a_concrete_fact_is_still_used(self):
        engine, _ = make_engine()
        add_fact(engine, "周五要交稿", importance=1.0)
        text, kind = engine.proactive.compose("u1")
        self.assertEqual("memory", kind)
        self.assertIn("周五要交稿", text)


class ProactiveHistoryTests(unittest.TestCase):
    def test_proactive_message_is_part_of_the_next_context(self):
        # The reported bug: the bot sent "突然想到，你叫蓝心心。" outside turns;
        # the user replied "?" and the model, unable to see its own message,
        # explained something it never said ("我说安静待着…").
        engine, _ = make_engine()
        engine.record_proactive_turn("u1", "突然想到，你叫“蓝心心”。")
        rows = engine.db.recent_turns("u1", 10)
        self.assertEqual(1, len(rows))
        self.assertEqual("assistant", rows[-1]["role"])
        self.assertEqual("突然想到，你叫“蓝心心”。", rows[-1]["content"])

        plan = engine.begin_turn("u1", "？")
        contents = [m["content"] for m in plan.recent_messages]
        self.assertIn("突然想到，你叫“蓝心心”。", contents)
        self.assertEqual("assistant", plan.recent_messages[-1]["role"])

    def test_empty_proactive_text_is_ignored(self):
        engine, _ = make_engine()
        engine.record_proactive_turn("u1", "   ")
        self.assertEqual([], engine.db.recent_turns("u1", 10))


if __name__ == "__main__":
    unittest.main()
