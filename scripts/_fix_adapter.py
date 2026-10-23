import pathlib
import sys

p = pathlib.Path(r"src\zhenxin\adapters\astrbot.py")
s = p.read_text(encoding="utf-8")
bad = "`n"
print("literal marker present:", bad in s)
s = s.replace(bad, "\n")
if "import sys" not in s.split("log = logging")[0]:
    s = s.replace("import os\n", "import os\nimport sys\n", 1)
p.write_text(s, encoding="utf-8")
sys.path.insert(0, "src")
import zhenxin.adapters.astrbot as m  # noqa: E402

print("import ok:", hasattr(m.AstrBotAdapter, "on_llm_request"))
