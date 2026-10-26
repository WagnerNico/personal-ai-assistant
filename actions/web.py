"""Read-only web search and bounded reading of public HTML/text pages."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from .web_sources import matches_source, source_domain
from .web_content import PageDetails, relevant_excerpt


class WebError(RuntimeError):
    """A web request could not be completed safely or yielded no useful text."""

    def __init__(self, message, *, code=""):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WebLink:
    title: str
    url: str


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    snippet: str = ""
    text: str = ""
    page_read: bool = False
    published_at: str = ""
    publisher: str = ""
    metadata: str = ""
    links: tuple[WebLink, ...] = ()


@dataclass(frozen=True)
class WebEvidence:
    sources: tuple[WebSource, ...] = ()
    error: str = ""
    error_code: str = ""


def public_url(url, *, resolve=True):
    """Reject credentials, local networks and non-web schemes on every hop."""
    try:
        if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 33 for c in url):
            raise ValueError()
        parsed = urlsplit(url)
        host = parsed.hostname
        if (parsed.scheme not in ("http", "https") or not host or parsed.username is not None
                or parsed.password is not None or parsed.port not in (None, 80, 443)
                or "\\" in url):
            raise ValueError()
        host = host.encode("idna").decode("ascii").rstrip(".")
        if host.casefold() == "localhost" or host.casefold().endswith((".localhost", ".local", ".internal")):
            raise ValueError()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        addresses = [str(literal)] if literal else []
        if resolve and not literal:
            addresses = list(dict.fromkeys(item[4][0] for item in
                                          socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
        for address in addresses:
            ip = ipaddress.ip_address(address)
            ip = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip
            if not ip.is_global or ip.is_multicast or ip.is_reserved:
                raise ValueError()
        if resolve and not addresses:
            raise ValueError()
        netloc = f"[{host}]" if ":" in host else host
        if parsed.port is not None:
            netloc += f":{port}"
        clean = urlunsplit((parsed.scheme, netloc, parsed.path or "/", parsed.query, ""))
        return clean, host, port, addresses
    except OSError as error:
        raise WebError("The website address could not be resolved. Check your internet connection.") from error
    except (ValueError, UnicodeError) as error:
        raise WebError("Only public HTTP or HTTPS pages can be read.") from error


class _PageText(HTMLParser):
    """Extract visible text without executing scripts or loading page resources."""

    HIDDEN = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = []
        self.in_title = False
        self.title = []
        self.parts = []
        self.main_parts = []
        self.article_parts = []
        self.main_depth = 0
        self.article_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.main_depth += 1
        if tag == "article":
            self.article_depth += 1
        if tag == "title":
            self.in_title = True
        if tag in self.HIDDEN:
            self.hidden.append(tag)
        if not self.hidden and tag in ("p", "div", "br", "li", "h1", "h2", "h3", "article"):
            self.parts.append("\n")
            if self.main_depth:
                self.main_parts.append("\n")
            if self.article_depth:
                self.article_parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "main":
            self.main_depth = max(0, self.main_depth - 1)
        if tag == "article":
            self.article_depth = max(0, self.article_depth - 1)
        if tag == "title":
            self.in_title = False
        if tag in self.hidden:
            # Malformed markup cannot leave a matching hidden element open.
            self.hidden = self.hidden[:len(self.hidden) - 1 - self.hidden[::-1].index(tag)]

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if not self.hidden:
            self.parts.append(data)
            if self.main_depth:
                self.main_parts.append(data)
            if self.article_depth:
                self.article_parts.append(data)

    def result(self):
        parts = self.article_parts or self.main_parts or self.parts
        lines = (" ".join(line.split()) for line in " ".join(parts).splitlines())
        return " ".join(self.title).strip()[:300], "\n".join(line for line in lines if line)[:18000]


class PublicPageReader:
    """Use GET only, with no cookies, authentication, proxies or saved downloads.

    Connections use the validated IP directly, preserving the original Host
    header and TLS certificate checks to avoid a second DNS resolution.
    """

    def __init__(self, *, timeout=6, max_bytes=2 * 1024 * 1024):
        self.timeout = timeout
        self.max_bytes = max_bytes

    def _connection(self, scheme, host, port, address):
        if scheme == "https":
            context = ssl.create_default_context()
            class Connection(http.client.HTTPSConnection):
                def connect(self):
                    sock = socket.create_connection((address, port), self.timeout)
                    try:
                        self.sock = context.wrap_socket(sock, server_hostname=host)
                    except BaseException:
                        sock.close()
                        raise
            return Connection(host, port, timeout=self.timeout, context=context)
        class Connection(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.create_connection((address, port), self.timeout)
        return Connection(host, port, timeout=self.timeout)

    def fetch(self, url, *, content_types=("text/html", "text/plain", "application/xhtml+xml"),
              user_agent=None, source_domain=None):
        """Retrieve bounded bytes for page and fixed-provider data adapters."""
        try:
            deadline = time.monotonic() + 12
            for _ in range(5):
                url, host, port, addresses = public_url(url)
                if source_domain and not matches_source(url, source_domain):
                    raise WebError("The page redirected outside the requested source.", code="source_redirect")
                parsed = urlsplit(url)
                target = quote(urlunsplit(("", "", parsed.path, parsed.query, "")), safe="/%?=&:+,;@!$'()*[]~-_")
                connection = self._connection(parsed.scheme, host, port, addresses[0])
                try:
                    connection.request("GET", target, headers={
                        "User-Agent": user_agent or "Airis/0.1 (public-page reader)",
                        "Accept": ", ".join(content_types),
                        "Accept-Encoding": "identity",
                    })
                    response = connection.getresponse()
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.getheader("Location")
                        if not location:
                            raise WebError("The page returned an invalid redirect.")
                        url = urljoin(url, location)
                        continue
                    if response.status != 200:
                        raise WebError("The website did not allow this page to be read.", code=f"http_{response.status}")
                    content_type = response.getheader("Content-Type", "").lower()
                    if not content_type.startswith(content_types):
                        raise WebError("The website returned an unsupported content type.")
                    if response.getheader("Content-Encoding", "identity").lower() != "identity":
                        raise WebError("The website returned an unsupported compressed page.")
                    chunks, size = [], 0
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise WebError("The page took too long to read.")
                        if connection.sock is not None:
                            connection.sock.settimeout(min(self.timeout, remaining))
                        chunk = response.read1(min(32768, self.max_bytes + 1 - size))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise WebError("The page is too large to read.")
                    body = b"".join(chunks)
                    return url, body, content_type
                finally:
                    connection.close()
            raise WebError("The page redirected too many times.")
        except (OSError, http.client.HTTPException) as error:
            raise WebError("The page could not be reached. Please try again later.") from error

    def read_json(self, url):
        """Decode a provider's public JSON using the same network boundaries."""
        _, body, _ = self.fetch(url, content_types=("application/json",))
        try:
            return json.loads(body)
        except (ValueError, RecursionError) as error:
            raise WebError("The data provider returned invalid JSON.") from error

    def read(self, url, *, source_domain=None):
        options = {"source_domain": source_domain} if source_domain else {}
        types = ("text/html", "text/plain", "application/xhtml+xml", "application/json", "application/ld+json")
        try:
            url, body, content_type = self.fetch(url, content_types=types, **options)
        except WebError as error:
            if error.code != "http_403":
                raise
            from .web_http import browser_fetch
            url, body, content_type = browser_fetch(url, timeout=self.timeout, max_bytes=self.max_bytes,
                                                   content_types=types, **options)
        host = urlsplit(url).hostname
        encoding = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type)
        try:
            html = body.decode(encoding[1] if encoding else "utf-8", errors="replace")
        except LookupError:
            html = body.decode("utf-8", errors="replace")
        metadata, links = "", ()
        if content_type.startswith(("application/json", "application/ld+json")):
            try:
                text = json.dumps(json.loads(html), ensure_ascii=False)[:10000]
            except (ValueError, RecursionError) as error:
                raise WebError("The website returned invalid JSON.") from error
            title = host
        elif content_type.startswith("text/plain"):
            title, text = host, html.strip()[:6000]
        else:
            parser = _PageText()
            parser.feed(html)
            title, text = parser.result()
            details = PageDetails(url)
            details.feed(html)
            metadata = details.metadata()
            candidates, seen = [], set()
            for label, target in details.links:
                try:
                    target, *_ = public_url(target, resolve=False)
                    if target not in seen and target != url and (not source_domain or matches_source(target, source_domain)):
                        candidates.append(WebLink(label, target))
                        seen.add(target)
                except WebError:
                    continue
                if len(candidates) == 12:
                    break
            links = tuple(candidates)
        if (not (text or metadata) or re.search(r"captcha|just a moment|access denied|consent|before you continue", title, re.I)
                or re.search(r"verify (?:that )?you are human", text, re.I)):
            raise WebError("The page requires browser access or contains no readable text.")
        return WebSource(title or host, url, text=text, page_read=True, metadata=metadata, links=links)


class DDGSSearch:
    """A replaceable, API-key-free search adapter, loaded only for web requests."""

    def _rss_news(self, query):
        """Read publisher-dated news feeds, then find original article links."""
        now = datetime.now(timezone.utc)
        reader = PublicPageReader()
        for days in (7, 31):
            if query.casefold() == "world politics":
                endpoint = "https://news.google.com/rss/headlines/section/topic/WORLD?" + urlencode(
                    {"hl": "en-US", "gl": "US", "ceid": "US:en"})
            else:
                endpoint = "https://news.google.com/rss/search?" + urlencode(
                    {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
            _, body, _ = reader.fetch(endpoint, content_types=("application/rss+xml", "application/xml", "text/xml"))
            try:
                root = ET.fromstring(body)
            except ET.ParseError as error:
                raise WebError("The news provider returned an invalid feed.") from error
            sources, seen = [], set()
            for item in root.findall("./channel/item"):
                try:
                    date = parsedate_to_datetime(item.findtext("pubDate", ""))
                    if not now - timedelta(days=days) <= date <= now + timedelta(hours=1):
                        continue
                    url, *_ = public_url(item.findtext("link", ""))
                    publisher = item.findtext("source", "")[:200]
                    title = item.findtext("title", "")[:300]
                    if publisher and title.endswith(" - " + publisher):
                        title = title[:-(len(publisher) + 3)]
                    if (not title or title.casefold() in seen
                            or re.search(r"stock price.*(?:quote|history)|latest.*breaking.*news", title, re.I)):
                        continue
                    seen.add(title.casefold())
                    parser = _PageText()
                    parser.feed(item.findtext("description", ""))
                    _, summary = parser.result()
                    sources.append(WebSource(title, url, snippet=summary[:1000] or title,
                                             published_at=date.isoformat(), publisher=publisher))
                except (ValueError, TypeError, WebError):
                    continue
            if sources:
                # The feed already ranks relevance; keep publisher variety
                # rather than filling the entire answer with one outlet.
                selected, counts = [], {}
                for source in sources:
                    if counts.get(source.publisher, 0) >= 2:
                        continue
                    counts[source.publisher] = counts.get(source.publisher, 0) + 1
                    selected.append(source)
                    if len(selected) == 6:
                        break
                sources = selected
                def original_link(source):
                    try:
                        constraint = re.search(r"\bsite:([\w.-]+)", query, re.I)
                        suffix = f" site:{constraint[1]}" if constraint else ""
                        matches = self.search(f"{' '.join(source.title.split()[:14])} {source.publisher}{suffix}")
                    except WebError:
                        return source
                    words = set(re.findall(r"\w{3,}", source.title.casefold()))
                    for match in matches:
                        overlap = len(words & set(re.findall(r"\w{3,}", match.title.casefold()))) / max(1, len(words))
                        if overlap >= 0.6 and _article_url(match.url):
                            return replace(source, url=match.url, snippet=match.snippet or source.snippet)
                    return source
                with ThreadPoolExecutor(max_workers=3) as executor:
                    sources[:3] = list(executor.map(original_link, sources[:3]))
                return sources
        raise WebError("No recent dated news articles were available.")

    def search(self, query):
        try:
            from ddgs import DDGS
        except ImportError as error:
            raise WebError("Web search is not available in this installation. Update Airis and try again.") from error
        rows = None
        for backends in ("duckduckgo,brave", "google,yahoo", "bing,mojeek"):
            try:
                rows = DDGS(timeout=6).text(query, max_results=4, backend=backends)
                if rows:
                    break
            except Exception:
                continue
        if not rows:
            raise WebError("Web search is unavailable. Please try again later.")
        sources = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                url, *_ = public_url(row.get("href", ""))
            except WebError:
                continue
            if url in seen:
                continue
            seen.add(url)
            sources.append(WebSource(str(row.get("title") or url)[:300], url,
                                     snippet=str(row.get("body") or "")[:700]))
            if len(sources) == 4:
                break
        if not sources:
            raise WebError("Web search returned no usable results. Try a more specific query.")
        return sources

    def news(self, query):
        """Get dated articles rather than category/quote landing pages."""
        try:
            return self._rss_news(query)
        except WebError:
            pass
        try:
            from ddgs import DDGS
        except ImportError as error:
            raise WebError("News search is not available in this installation.") from error
        now = datetime.now(timezone.utc)
        for window, days in (("w", 7), ("m", 31)):
            try:
                rows = DDGS(timeout=6).news(query, max_results=8, timelimit=window, backend="yahoo,bing")
            except Exception:
                continue
            sources, seen = [], set()
            for row in rows:
                try:
                    published = datetime.fromisoformat(str(row.get("date", "")).replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                    if not now - timedelta(days=days) <= published <= now + timedelta(hours=1):
                        continue
                    url, *_ = public_url(row.get("url", ""))
                    if not _article_url(url):
                        continue
                    dated_path = re.search(r"/(20\d{2})/(\d{2})(?:/(\d{2}))?/", urlsplit(url).path)
                    if dated_path:
                        path_date = datetime(int(dated_path[1]), int(dated_path[2]), int(dated_path[3] or 1), tzinfo=timezone.utc)
                        if path_date < now - timedelta(days=days):
                            continue
                    if url in seen:
                        continue
                    seen.add(url)
                    sources.append(WebSource(str(row.get("title") or url)[:300], url,
                                             snippet=str(row.get("body") or "")[:1000],
                                             published_at=published.isoformat(),
                                             publisher=str(row.get("source") or "")[:200]))
                except (ValueError, TypeError, AttributeError, WebError):
                    continue
            if sources:
                sources.sort(key=lambda source: source.published_at, reverse=True)
                return sources[:6]
        raise WebError("No recent dated news articles could be retrieved. I couldn't verify the latest headlines.")


class WebAccess:
    """Fetch evidence; never pass remote text to an action executor."""

    def __init__(self, *, search=None, reader=None):
        self.search_provider = search if search is not None else DDGSSearch()
        self.reader = reader if reader is not None else PublicPageReader()
        self._source_domains = {}

    def _source_domain(self, source, on_stage):
        if source in self._source_domains:
            return self._source_domains[source]
        try:
            return source_domain(source)
        except WebError as error:
            if error.code != "source_identity":
                raise
        if on_stage:
            on_stage("Finding the requested source...")
        words = [word for word in re.findall(r"\w+", source.casefold())
                 if word not in {"the", "official", "website", "news"}]
        if not words:
            raise WebError("Please provide the source's website address.")
        brand = "".join(words)
        identities = {brand, "the" + brand, brand + "news", "the" + brand + "news"}
        # Prefer the highest-ranked public homepage whose host identifies the
        # named publisher. Do not let an arbitrary article mention select it.
        for candidate in self.search_provider.search(f"{source} official website"):
            try:
                _, host, _, _ = public_url(candidate.url, resolve=False)
                identity = re.sub(r"[^\w]", "", host.casefold().removeprefix("www.").split(".")[0])
                if urlsplit(candidate.url).path in ("", "/") and identity in identities:
                    domain = source_domain(candidate.url)
                    if len(self._source_domains) >= 32:
                        self._source_domains.pop(next(iter(self._source_domains)))
                    self._source_domains[source] = domain
                    return domain
            except WebError:
                continue
        raise WebError(f"I couldn't identify the website for {source}. Please provide its website address.")

    def gather(self, *, query=None, url=None, kind="search", on_stage=None, source=None):
        if (query is None) == (url is None):
            raise ValueError("Provide one search query or one page URL.")
        try:
            domain = self._source_domain(source, on_stage) if source else None
            if url is not None:
                if on_stage:
                    on_stage("Reading web page...")
                if domain and not matches_source(url, domain):
                    raise WebError("The page is outside the requested source.")
                page = self.reader.read(url, **({"source_domain": domain} if domain else {}))
                if domain and not matches_source(page.url, domain):
                    raise WebError("The page redirected outside the requested source.")
                return WebEvidence((page,))
            if not isinstance(query, str) or not query.strip() or len(query) > 500:
                raise WebError("Please use a nonempty search query of at most 500 characters.")
            if kind not in {"search", "news"}:
                raise WebError("Unsupported read-only web request.")
            if on_stage:
                on_stage("Searching the web...")
            # The constraint is applied to the query, returned links and every
            # page redirect. A source keyword alone is not an origin guarantee.
            search_query = re.sub(r"\bsite:\S+", "", query, flags=re.I).strip() if domain else query.strip()
            if domain:
                search_query += f" site:{domain}"
            if kind == "news":
                sources = list(self.search_provider.news(search_query))[:6]
            else:
                sources = list(self.search_provider.search(search_query))[:4]
            if domain:
                sources = [item for item in sources if matches_source(item.url, domain)]
                if not sources:
                    raise WebError(f"No usable results were found on {domain}. I have not substituted another source.")
            if on_stage:
                on_stage("Reading search results...")
            def read_source(source):
                if urlsplit(source.url).hostname == "news.google.com":
                    return source  # Opaque feed links are not article bodies.
                try:
                    page = self.reader.read(source.url, **({"source_domain": domain} if domain else {}))
                    if domain and not matches_source(page.url, domain):
                        return source
                    return replace(page, title=source.title if kind == "news" else page.title,
                                   text=relevant_excerpt(page.text, query), snippet=source.snippet,
                                   published_at=source.published_at, publisher=source.publisher)
                except WebError:
                    return source  # Search snippets remain useful but are labeled.
            with ThreadPoolExecutor(max_workers=2) as executor:
                sources[:2] = list(executor.map(read_source, sources[:2]))
                if sum(source.page_read for source in sources[:2]) < 2 and len(sources) > 2:
                    sources[2:4] = list(executor.map(read_source, sources[2:4]))
            if not sources:
                raise WebError("Web search returned no usable results.")
            return WebEvidence(tuple(sources))
        except WebError as error:
            return WebEvidence(error=str(error), error_code=error.code)


def _article_url(url):
    path = urlsplit(url).path.strip("/").casefold()
    return bool(path and path not in ("news", "politics", "world", "news/politics", "news/world")
                and not re.search(r"(?:^|/)(?:quote|markets/companies)(?:/|$)|company-group", path))
