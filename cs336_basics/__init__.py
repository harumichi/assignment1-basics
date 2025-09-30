import importlib.metadata
import logging

__version__ = importlib.metadata.version("cs336_basics")

# シンプルなロギング初期化: 既存ハンドラが無ければ basicConfig、あればレベルを INFO まで引き下げ。
_root = logging.getLogger()
if not _root.handlers:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
elif _root.level > logging.INFO:
	_root.setLevel(logging.INFO)

# パッケージロガーはプロパゲートさせつつ追加設定不要。
