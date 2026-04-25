import ssl
import logging
from requests import Session
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from typing import Dict, Optional, Any
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager, ProxyManager


retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504, 104],
    allowed_methods=["HEAD", "POST", "PUT", "GET", "OPTIONS"],
)

logger = logging.getLogger("Browser")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


class CipherSuiteAdapter(HTTPAdapter):
    """
    A Sonar-compliant Transport Adapter that handles custom SSL contexts,
    cipher suites, and ECDH curves without monkey-patching wrap_socket.
    """

    def __init__(self, *args, **kwargs):

        self.cipherSuite = kwargs.pop("cipherSuite", None)
        self.ecdhCurve = kwargs.pop("ecdhCurve", "prime256v1")
        self.server_hostname = kwargs.pop("server_hostname", None)
        self.ssl_context = kwargs.pop("ssl_context", None)
        super().__init__(**kwargs)

    def _create_hardened_context(self) -> ssl.SSLContext:
        """Create a high-security SSL context that satisfies Sonar requirements."""
        # Purpose.SERVER_AUTH loads system CA certs and enables verification
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # Enforce modern TLS (Satisfies Sonar S4423)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Explicitly disable legacy insecure features
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_COMPRESSION

        if self.cipherSuite:
            context.set_ciphers(self.cipherSuite)

        if self.ecdhCurve and hasattr(context, "set_ecdh_curve"):
            context.set_ecdh_curve(self.ecdhCurve)

        return context

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        """Native way to inject ssl_context into urllib3."""
        pool_kwargs["ssl_context"] = self.ssl_context
        # If server_hostname is provided, it's used for SNI
        if self.server_hostname:
            pool_kwargs["server_hostname"] = self.server_hostname

        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, **pool_kwargs
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Ensure the hardened context is also used for proxy connections."""
        proxy_kwargs["ssl_context"] = self.ssl_context
        if self.server_hostname:
            proxy_kwargs["server_hostname"] = self.server_hostname

        self.proxy_manager[proxy] = ProxyManager(proxy, **proxy_kwargs)
        return self.proxy_manager[proxy]


class Browser(Session):

    def __init__(
        self,
        proxies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        cipher_suite: Optional[str] = None,
        ecdh_curve: str = "prime256v1",
        server_hostname: Optional[str] = None,
        debug: bool = False,
    ):
        super().__init__()
        if proxies:
            self.proxies.update(proxies)
        self.headers.update(headers or {})
        self.debug = debug
        self.response = None

        # Initialize the hardened adapter
        adapter = CipherSuiteAdapter(
            cipherSuite=cipher_suite,
            ecdhCurve=ecdh_curve,
            server_hostname=server_hostname,
            max_retries=retry_strategy,
        )

        # Mount to all https traffic
        self.mount("https://", adapter)

    def get_soup(self):
        if self.response is None:
            raise RuntimeError("No response available")
        if not self.response.ok:
            raise RuntimeError(self.response.reason)
        return BeautifulSoup(self.response.content, "html.parser")

    def get_json(self):
        if self.response is None:
            raise RuntimeError("No response available")
        if not self.response.ok:
            raise RuntimeError(self.response.reason)
        try:
            return self.response.json()
        except Exception:
            return None

    def send_request(
        self, method: str, url: str, headers: Optional[Dict[str, str]] = None, **kwargs
    ):
        # Create a copy of headers safely
        merged_headers = dict(self.headers)
        if headers:
            merged_headers.update(headers)

        if self.proxies:
            kwargs["proxies"] = self.proxies

        logger.debug("Using proxies: %s", self.proxies)

        self.response = self.request(
            method,
            url,
            headers=merged_headers,
            **kwargs,
        )

        if self.debug and self.response is not None:
            logger.debug(f"→ {method} {url}")
            logger.debug(f"Status: {self.response.status_code}")
            logger.debug(f"Headers enviados: {merged_headers}")
            logger.debug(f"Headers recebidos: {dict(self.response.headers)}")
            content_preview = self.response.text[:250].strip().replace("\n", "")
            logger.debug(f"Body (preview): {content_preview} [...]")

        return self.response
