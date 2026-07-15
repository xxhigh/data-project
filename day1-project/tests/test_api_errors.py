import asyncio

import httpx
import pytest

from main import ApiCallError, fetch_url


def test_fetch_url_raises_api_call_error_on_http_status() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(ApiCallError, match="HTTP 500 response"):
                await fetch_url(client, "https://example.test/fail")

    asyncio.run(run())


def test_fetch_url_raises_api_call_error_on_request_error() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(ApiCallError, match="connection failed"):
                await fetch_url(client, "https://example.test/fail")

    asyncio.run(run())
