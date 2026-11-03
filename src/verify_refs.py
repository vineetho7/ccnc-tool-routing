"""Verify every arXiv id in refs.bib against the real arXiv API; fetch authors + true title."""
import re, urllib.request, xml.etree.ElementTree as ET, time, json

bib = open("paper/refs.bib").read()
ids = re.findall(r"arXiv:(\d{4}\.\d{4,5})", bib)
keys = re.findall(r"@\w+\{(\w+),", bib)
print(f"{len(ids)} arXiv ids found\n")
NS = {"a": "http://www.w3.org/2005/Atom"}
out = {}
for aid in ids:
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    try:
        with urllib.request.urlopen(url, timeout=30) as f:
            root = ET.fromstring(f.read())
        e = root.find("a:entry", NS)
        title = " ".join((e.find("a:title", NS).text or "").split())
        if "Error" in title or not e.find("a:id", NS).text.endswith(tuple("0123456789")):
            raise ValueError("no entry")
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        published = (e.find("a:published", NS).text or "")[:4]
        out[aid] = {"title": title, "authors": authors, "year": published}
        print(f"{aid}  OK  {published}  {title[:72]}")
        print(f"          {len(authors)} authors, first: {authors[0] if authors else '?'}")
    except Exception as ex:
        out[aid] = None
        print(f"{aid}  *** NOT FOUND / ERROR: {ex}")
    time.sleep(3)
json.dump(out, open("results/ref_verification.json", "w"), indent=1)
