"""Browser-compatible public GET requests with validated IPs and manual redirects."""

import time
from urllib.parse import urljoin


def browser_fetch(url, *, timeout, max_bytes, content_types, source_domain=None):
    """Retry a blocked HTML reader without cookies, proxies or JavaScript."""
    from .web import WebError, public_url
    from .web_sources import matches_source
    try:
        from curl_cffi import CurlOpt, requests
    except ImportError as error:
        raise WebError("The browser-compatible page reader is not installed.") from error
    deadline = time.monotonic() + 12
    try:
        for _ in range(5):
            url, host, port, addresses = public_url(url)
            if source_domain and not matches_source(url, source_domain):
                raise WebError("The page redirected outside the requested source.", code="source_redirect")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WebError("The page took too long to read.")
            address = addresses[0]
            address = f"[{address}]" if ":" in address else address
            chunks, size, failure = [], 0, ""
            def collect(chunk):
                nonlocal size, failure
                size += len(chunk)
                if size > max_bytes:
                    failure = "The page is too large to read."
                    return 0
                if time.monotonic() > deadline:
                    failure = "The page took too long to read."
                    return 0
                chunks.append(chunk)
                return len(chunk)
            # RESOLVE pins the connection while preserving Host and TLS SNI.
            with requests.Session(impersonate="chrome", trust_env=False, verify=True,
                                  allow_redirects=False, discard_cookies=True,
                                  curl_options={CurlOpt.RESOLVE: [f"{host}:{port}:{address}"],
                                                CurlOpt.PROXY: "",
                                                CurlOpt.TIMEOUT_MS: max(1, int(remaining * 1000))}) as session:
                try:
                    response = session.get(url, content_callback=collect, timeout=min(timeout, remaining),
                                           headers={"Accept": ", ".join(content_types)},
                                           allow_redirects=False, discard_cookies=True)
                except Exception as error:
                    if failure:
                        raise WebError(failure) from error
                    raise
                try:
                    if failure:
                        raise WebError(failure)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise WebError("The page returned an invalid redirect.")
                        url = urljoin(url, location)
                        continue
                    if response.status_code != 200:
                        raise WebError("The website did not allow this page to be read.", code=f"http_{response.status_code}")
                    content_type = response.headers.get("content-type", "").lower()
                    if not content_type.startswith(content_types):
                        raise WebError("The website returned an unsupported content type.")
                    return url, b"".join(chunks), content_type
                finally:
                    response.close()
        raise WebError("The page redirected too many times.")
    except WebError:
        raise
    except Exception as error:
        raise WebError("The page could not be reached. Please try again later.") from error
