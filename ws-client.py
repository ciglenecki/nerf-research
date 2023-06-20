#!/usr/bin/env python
import sys
import asyncio
from websockets.sync.client import connect

def hello():
    with connect(sys.argv[1]) as websocket:
        websocket.send("Hello world!")
        message = websocket.recv()
        print(f"Received: {message}")

hello()
