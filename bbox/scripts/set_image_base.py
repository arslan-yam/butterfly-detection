import re
import sys

base = sys.argv[1].rstrip("/")
path = "../project/instruction.html"

html = open(path, encoding="utf-8").read()
html, n = re.subn(r'src="[^"]*/(\d+\.png)"', lambda m: f'src="{base}/{m.group(1)}"', html)
open(path, "w", encoding="utf-8").write(html)

print(f"replaced links: {n}")
