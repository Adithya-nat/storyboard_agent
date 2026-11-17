"""Loopback HTTPS (CONNECT) proxy for the hosted copy on a Mac.

Docker Desktop's outbound network path corrupts some large TLS uploads (Google sends back
"bad record mac"), while the container's traffic to the Mac's own loopback is clean. So the
api container sends HTTPS through this proxy, which opens the real connection from macOS.
TLS stays end to end; this only relays bytes. It listens on 127.0.0.1 and relays port 443 only.

    python3 deploy/egress_proxy.py [port]     # `make host-up` runs it under launchd
"""

import asyncio
import sys

HEADER_TIMEOUT = 10
CHUNK = 65536


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                break
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()  # half-close: let the other direction finish
    except (ConnectionError, OSError):
        writer.close()


async def _handle(client_r, client_w):
    upstream_w = None
    try:
        line = await asyncio.wait_for(client_r.readline(), HEADER_TIMEOUT)
        while (await asyncio.wait_for(client_r.readline(), HEADER_TIMEOUT)) not in (b"\r\n", b"\n", b""):
            pass
        method, target, _ = line.decode("latin-1").split()
        host, _, port = target.rpartition(":")
        if method != "CONNECT" or port != "443" or not host:
            client_w.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            upstream_r, upstream_w = await asyncio.wait_for(asyncio.open_connection(host.strip("[]"), 443), 15)
        except (OSError, asyncio.TimeoutError):
            client_w.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        client_w.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_w.drain()
        await asyncio.gather(_pipe(client_r, upstream_w), _pipe(upstream_r, client_w))
    except (ValueError, asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        for w in (client_w, upstream_w):
            if w is not None:
                w.close()


async def _main(port):
    server = await asyncio.start_server(_handle, "127.0.0.1", port)
    print(f"egress proxy on 127.0.0.1:{port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main(int(sys.argv[1]) if len(sys.argv) > 1 else 18443))
