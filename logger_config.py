import logging
from pathlib import Path

# 統一的日誌配置
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'app.log'  # 統一日誌文件

# 只配置一次，避免重複配置
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,  # 統一使用 INFO 級別（包含 ERROR）
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # 不輸出到終端，避免 AI 看到
        ]
    )

def get_logger(name):
    """獲取指定名稱的 logger"""
    return logging.getLogger(name)

