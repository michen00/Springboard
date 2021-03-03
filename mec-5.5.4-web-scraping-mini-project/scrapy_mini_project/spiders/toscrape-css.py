import scrapy
import sqlite3
from json import load
from collections import OrderedDict


class AuthorBlurbsSpider(scrapy.Spider):
    name = "toscrape-css"
    start_urls = [
        "http://quotes.toscrape.com/page/1/",
    ]
    biopages = set()

    def parse(self, response):
        def resp(x):
            return response.css(x)

        AuthorBlurbsSpider.biopages = AuthorBlurbsSpider.biopages.union(
            resp(".quote span a::attr(href)").getall()
        )
        next_page = resp("li.next a::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)
        else:
            # Author pages are parsed after all the links have been gathered.
            AuthorBlurbsSpider.biopages = [
                "".join([url, "/"]) for url in AuthorBlurbsSpider.biopages
            ]
            yield response.follow(
                AuthorBlurbsSpider.biopages.pop(),
                callback=self.parse_authorpages,
            )

    def parse_authorpages(self, response):
        def resp(x):
            return response.css(x)

        author = resp(".author-title::text").get().strip()
        yield {
            "author": author,
            "born": self.get_born(resp),
            "blurb": self.get_blurb(resp(".author-description::text"), author),
        }
        if AuthorBlurbsSpider.biopages:
            yield response.follow(
                self.biopages.pop(),
                callback=self.parse_authorpages,
            )
        else:
            del AuthorBlurbsSpider.biopages

    def get_blurb(self, bio, author):
        """Parses full author bio for an abridged blurb."""
        # E.g.: Author_Last_Name is/was [some text].
        blurb = bio.re_first(rf"{author.split()[-1]} (?:wa|i)s.+?\.")
        if blurb:
            return blurb
        # Otherwise, return the first two well-formed sentences
        # This may be <2 sentences for those with decimals or abbreviations
        # May return more if the bio isn't properly delimited (. )
        blurb = bio.re_first(r"(\b.+?\. .+?\.) ")
        return blurb if blurb else "blurb extraction failed"

    def get_born(self, resp):
        try:
            return int(resp(".author-born-date::text").get()[-4:])
        except Exception:
            return "malformed born date"

    def closed(self, reason):
        with open("css-scraper-results.json", "r") as f:
            data = load(f)
        try:
            conn = sqlite3.connect("css_scrapy_practice.db")
            c = conn.cursor()
            fields = OrderedDict(
                [
                    ("author", "TEXT"),
                    ("born", "INTEGER"),
                    ("blurb", "TEXT"),
                ]
            )
            c.execute("drop table if exists quotes")

            # querymaker(before_paren: str, within_paren: list[str]) -> str:
            def querymaker(before_paren: str, within_paren: list) -> str:
                return " ".join(
                    (
                        before_paren,
                        "".join(("(", ", ".join(within_paren), ");")),
                    )
                )

            c.execute(
                querymaker(
                    "create table quotes",
                    [" ".join((k, v)) for k, v in fields.items()],
                )
            )
            c.executemany(
                querymaker(
                    "insert into quotes values",
                    ["?"] * len(fields),
                ),
                [
                    [
                        int(record[field])
                        if fields[field] == "INTEGER"
                        else record[field]
                        for field in fields
                    ]
                    for record in data
                ],
            )
            conn.commit()
        except Exception as e:
            print(e)
        finally:
            if conn:
                conn.close()
