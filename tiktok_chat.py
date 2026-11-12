
import asyncio
import threading


class TikTokChatListener:
    def __init__(self, username, queue, status_callback):
        self.username = username.lstrip("@")
        self.queue = queue
        self.status_callback = status_callback
        self.thread = None
        self.client = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="tiktok-live-chat")
        self.thread.start()

    def _run(self):
        try:
            from TikTokLive import TikTokLiveClient
            from TikTokLive.events import CommentEvent, ConnectEvent
        except ImportError:
            self.status_callback("TikTokLive not installed")
            return
        try:
            self.client = TikTokLiveClient(unique_id=f"@{self.username}")

            @self.client.on(ConnectEvent)
            async def on_connect(_event):
                self.status_callback(f"connected to @{self.username}")

            @self.client.on(CommentEvent)
            async def on_comment(event):
                self.queue.append({
                    "comment": getattr(event, "comment", ""),
                    "user": getattr(getattr(event, "user", None), "unique_id", "anonymous"),
                })

            asyncio.run(self.client.connect())
        except Exception as exc:
            self.status_callback(f"no tiktok :< : {str(exc)[:50]}")

    def stop(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
