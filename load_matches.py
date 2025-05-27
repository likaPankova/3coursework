import os, subprocess, pandas as pd, logging, sys, warnings, requests, re, time
from bs4 import BeautifulSoup

FORCE_CRAWL  = False
CSV_FILE     = "matches_raw.csv"
DROPBOX_URL  = ("https://www.dropbox.com/scl/fi/zk0zb58etdun5opkkucux/"
                "matches_raw.csv?rlkey=nu08nqynhu4odrjslvbf790jb&st=eycqa32t&dl=1")  # dl=1 → direct download

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format="[{levelname:^7s} {asctime}] {message}",
                    datefmt="%H:%M:%S",
                    style='{')
logg = logging.getLogger("loader")
log  = lambda m, lvl="info": (getattr(logg, lvl)(m), print(m))[1]

if not FORCE_CRAWL:
    if not os.path.isfile(CSV_FILE):
        log(f"CSV not found locally — скачиваем из Dropbox …")
        try:
            subprocess.run(["wget", "-q", "-O", CSV_FILE, DROPBOX_URL], check=True)
            log("CSV успешно скачан.")
        except Exception as e:
            log(f"Не удалось скачать CSV: {e}", "error")

    if os.path.isfile(CSV_FILE):
        try:
            matches_raw = pd.read_csv(CSV_FILE)
            log(f"✓  Загружено {len(matches_raw)} матчей из {CSV_FILE}")
        except Exception as e:
            log(f"⚠  CSV повреждён ({e}) — придётся скрапить.")
            FORCE_CRAWL = True
    else:
        FORCE_CRAWL = True  # файл так и не появился

if FORCE_CRAWL:
    log("↻  Запускаем веб-краулер (это займёт пару минут).")

    BASE = "https://soccer365.ru"
    HEADERS = {"User-Agent":
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"}
    SEASON_URLS = {
        "2024-2025": f"{BASE}/competitions/19/results/",
        "2023-2024": f"{BASE}/competitions/19/2023-2024/results/",
        "2022-2023": f"{BASE}/competitions/19/2022-2023/results/",
    }

    def get_soup(url):
        log(f"GET  {url}")
        r = requests.get(url, headers=HEADERS, timeout=20); r.raise_for_status()
        if "Лига чемпионов УЕФА" not in r.text:
            raise ValueError("Wrong page")
        return BeautifulSoup(r.text, "lxml")

    def links(page):  # extract all match links from results page
        s = get_soup(page)
        return sorted({BASE+a["href"] for a in s.select("a.game_link[href]") if "/games/" in a["href"]})

    def parse(url):
        s = get_soup(url)
        l, r = s.select_one("div.live_game.left"), s.select_one("div.live_game.right")
        return dict(URL=url,
                    HomeTeam=l.a.text.strip(), HomeGoals=int(l.span.text.strip()),
                    AwayTeam=r.a.text.strip(), AwayGoals=int(r.span.text.strip()))

    rows = []
    for season, url in SEASON_URLS.items():
        log(f"\n════ {season} ════")
        for i, u in enumerate(links(url), 1):
            log(f"[{i:03}] {u}")
            try: rows.append(parse(u))
            except Exception as e: warnings.warn(f"{u} -> {e}")
    matches_raw = pd.DataFrame(rows)
    matches_raw.to_csv(CSV_FILE, index=False)
    log(f"Скрап завершён: {len(matches_raw)} матчей сохранено в {CSV_FILE}")

matches = matches_raw[["HomeTeam","AwayTeam","HomeGoals","AwayGoals"]]
print(matches.head())