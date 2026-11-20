"""Build local dark HTML reading copies of the three research repositories using Pandoc."""
import html
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Research documents"
REPOSITORIES = ("fixed-reference-benchmark", "parallax-depth-benchmark", "routed-expert-splice")
STYLE = """
:root { color-scheme: dark; background: #101916; color: #e5eee9 }
* { box-sizing: border-box }
body { max-width: 84ch; margin: auto; padding: 2.5rem clamp(1.1rem, 5vw, 3.5rem) 5rem;
       font: 17px/1.65 system-ui, sans-serif; overflow-wrap: anywhere }
h1, h2, h3 { line-height: 1.2; text-wrap: balance }
h1, h2 { font-family: Georgia, serif; font-weight: 500 }
h1 { font-size: clamp(2rem, 4.5vw, 3.1rem) }
h2 { margin-top: 2rem; font-size: 1.7rem }
a { color: #84dfc2; text-underline-offset: .2em }
a:visited { color: #cbb2ef }
a:focus-visible { outline: 2px solid #8de4c5; outline-offset: 3px }
nav { font: 600 .875rem system-ui, sans-serif }
nav a { color: #bad2c6 }
code, pre { background: #1d2b24; color: #e7eee9; border-radius: .25rem }
code { padding: .08em .22em; font-size: .9em }
pre { padding: 1rem; overflow-x: auto }
pre code { padding: 0; overflow-wrap: normal }
blockquote { margin-left: 0; padding-left: 1rem; border-left: 3px solid #648c78; color: #c7d9ce }
table { display: block; overflow-x: auto; border-collapse: collapse; font-size: .92em }
td, th { text-align: left; border-bottom: 1px solid #43574b; padding: .5rem .75rem }
th { color: #d0e4d7; background: #203229 }
img { max-width: 100%; height: auto }
hr { border: 0; border-top: 1px solid #43574b }
@media print { :root { print-color-adjust: exact; -webkit-print-color-adjust: exact }
               body { max-width: none } nav { display: none } }
"""

def write_page(path, title, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    page = (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="color-scheme" content="dark">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{STYLE}</style><body>{content}</body></html>\n')
    tmp = path.with_suffix(".tmp")
    tmp.write_text(page)
    tmp.replace(path)

def relative(target, page):
    return quote(os.path.relpath(target, page.parent), safe="/")

def render(source, page, outputs):
    fragment = subprocess.check_output(["pandoc", "--from=gfm", "--to=html5", "--wrap=none",
                                        "--syntax-highlighting=none", str(source)], text=True)
    def link(match):
        attr, raw = match.groups()
        u = urlsplit(html.unescape(raw))
        if u.scheme or u.netloc or not u.path or u.path.startswith("/"):
            return match.group()
        target = (source.parent / unquote(u.path)).resolve()
        if target.is_dir() and (target / "README.md") in outputs:
            target = target / "README.md"
        url = urlunsplit(("", "", relative(outputs.get(target, target), page), u.query, u.fragment))
        return f'{attr}="{html.escape(url, quote=True)}"'
    fragment = re.sub(r'\b(href|src)="([^"]*)"', link, fragment)
    h = re.search(r"<h1\b[^>]*>(.*?)</h1>", fragment, re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", h.group(1))) if h else source.stem
    nav = f'<nav><a href="{relative(OUTPUT / "index.html", page)}">Research documents</a></nav>'
    write_page(page, title, nav + '<main>' + fragment + '</main>')
    return title

def main():
    outputs = {}
    for name in REPOSITORIES:
        repo = ROOT / name
        tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z",
                                           "--cached", "--others", "--exclude-standard", "*.md"], text=True)
        for entry in filter(None, tracked.split("\0")):
            source = repo / entry
            if source.is_file() and source.name != "AGENTS.md":
                outputs[source] = OUTPUT / source.relative_to(ROOT).with_suffix(".html")
    index = OUTPUT / "index.html"
    sections = {repo: [] for repo in REPOSITORIES}
    for source, page in outputs.items():
        title = render(source, page, outputs)
        repo = source.relative_to(ROOT).parts[0]
        sections[repo].append(f'<li><a href="{relative(page, index)}">{html.escape(title)}</a></li>')
    contents = ''.join(f"<section><h2>{repo}</h2><ul>{''.join(rows)}</ul></section>"
                       for repo, rows in sections.items())
    write_page(index, "Research documents", "<main><h1>Research documents</h1>" + contents + "</main>")
    print(f"Rendered {len(outputs)} documents in {OUTPUT}")

if __name__ == "__main__":
    main()
