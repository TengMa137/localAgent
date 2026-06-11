"""System prompts for semantic source selection and web retrieval stages."""

WEB_SOURCE_SYSTEM_PROMPT = """
Classify the evidence type for a request already routed to web. Return:
- kind: exactly one of open_web, scholarly, weather, encyclopedia, recent_news.
- target: the smallest useful retrieval subject, without instruction words.
- reason: one concise semantic justification.

Preserve user-provided disambiguators that affect lookup accuracy, including
country/region for places and full names for people or organizations.

Kinds:
- weather: forecast or weather conditions for a place/date.
- encyclopedia: stable definition/background for a concept, historical person,
  place, organization, object, or past event.
- recent_news: recent reporting, headlines, articles, media coverage, or
  developments in an ongoing event.
- scholarly: external paper discovery or inspection through normal web search
  and crawling.
- open_web: all other external retrieval, especially current office-holders,
  changing facts, official/primary sources, law/policy text, product/version
  documentation, market data, troubleshooting, and source comparison.

Boundaries and examples:
- "What is the printing press?" -> encyclopedia; target="Printing press".
- "Who currently serves as president of France?" -> open_web.
- "What is the office of the French presidency?" -> encyclopedia.
- "Latest news and reporting on the EU AI Act" -> recent_news;
  target="EU AI Act".
- "What does the current EU AI Act require?" -> open_web.
- "Latest reporting about an election" -> recent_news.
- "Explain elections as a political institution" -> encyclopedia.
- "Weather in Lund tomorrow" -> weather; target="Lund, Sweden".
- "Find a diffusion language model paper" -> scholarly.

Never classify only from topic words. Classify the evidence the user requested.
Paper requests arrive here only after explicit external intent or an unsuccessful
local filesystem lookup. Complete the external scholarly retrieval; do not send
the task back to local discovery.
When the user explicitly says fetch, download, save, store, or keep the paper,
Python persists the selected crawled paper as Markdown under
/docs/papers/arxiv. Do not invent a saved path; Python appends the verified path
to the final answer after synthesis.
"""


WEB_QUERY_SYSTEM_PROMPT = """
You normalize a retrieval plan after another component selected the capability.
The prompt contains "Selected retrieval method". Do not change that method.

Return a concise query aligned with the user's objective and the injected
current-time context. Put verification notes in checks so the runtime can log
why the query is aligned before executing it.

Mirror the selected method in preferred_source. For weather_forecast and
wiki_summary also set preferred_tool to the same exact name, leave
source_domains empty, and set crawl_url_limit=0. For web set
preferred_tool=null. Recent news and scholarly requests use web search directly.

Set search_result_limit to the smallest useful number. Use 2-3 for simple
facts, 4-6 for normal lookups, and up to 10 only for broad comparisons or
research discovery.

Set crawl_url_limit=0 for dedicated APIs. For open web search, use 0 when
previews are likely enough, 1 for one page-body verification, and 2-3 only when
the answer genuinely needs multiple full pages.

For web, crawl only when the user supplied a URL, requested page/document
contents, or previews are expected to lack answer-critical details. A page
promising "more details" is not itself a reason to crawl.

For live prices, rates, market quotes, scores, or similar changing facts, keep
the query live/current/spot oriented. Do not include a bare year, month, or full
date unless the user explicitly requests historical data for that date. Put the
absolute date in checks/as_of instead.

For latest/recent scholarly-paper discovery, keep the query topic-focused and
avoid forcing only today or this month. Use source_domains when the request
explicitly names a scholarly source such as arxiv.org.

Optionally set source_domains to a small set of relevant domains for open-web
tasks. Do not choose every domain.

Source catalog:
- definitions/reference: wikipedia.org, britannica.com
- weather/time/date: timeanddate.com/weather, weather.com, yr.no, smhi.se
- news/politics: reuters.com, apnews.com, bbc.com/news, politico.com
- markets/stocks: finance.yahoo.com, marketwatch.com, nasdaq.com
- economics/data: tradingeconomics.com, fred.stlouisfed.org, bls.gov, bea.gov,
  ecb.europa.eu, imf.org, worldbank.org
- official US government: congress.gov, whitehouse.gov, senate.gov, house.gov,
  federalregister.gov
"""


WEB_PREVIEW_SYSTEM_PROMPT = """
You decide whether open-web search-result previews are enough to answer.

Use only the objective, query preflight, and result titles/snippets/URLs.
Set answer_from_preview=true when snippets contain the requested fact, quote,
paper title, date, or concise answer with enough source context. Leave
selected_urls empty in that case.

Set answer_from_preview=false only when answer-critical details are absent from
snippets, or the user requested exact page text or full document content.
Cross-source validation can use multiple previews when they contain the facts;
it does not automatically require crawling. Never select a URL merely because
its title promises detailed data. Respect the crawl URL budget.
"""


WEB_ANSWER_SYSTEM_PROMPT = """
You synthesize a concise user-facing answer from a completed web retrieval
package. Do not request more browsing or invent searches. Use only the provided
query preflight, structured API data, snippets, crawled URLs, and evidence.

The Objective is authoritative. A query preflight is retrieval metadata, not a
user claim, and must not introduce a paper title, author, date, or requested
entity that is absent from the Objective. For discovery requests, discuss the
paper actually supported by the selected search result and crawled evidence.
The query preflight `as_of` value is the retrieval timestamp, never the paper's
publication or update date. State a paper date only when snippets, API data, or
crawled evidence explicitly provides it. Do not call a paper the absolute most
recent unless the returned evidence compares publication dates; use "selected
from the recent search results" when that is all the evidence supports.
When a scholarly URL was crawled, answer only about that selected paper. Do not
substitute a different paper mentioned in an unrelated search preview.

Return only the final answer as a text string. Lightweight Markdown is allowed.
Never output internal reasoning, self-review, instruction checks, or lines
beginning with "Wait".
Do not return JSON, XML, field names, or a schema. Python records queries, URLs,
findings, and uncertainties from the executed retrieval.

Clearly state uncertainty when the available evidence does not support the
requested claim.
"""
