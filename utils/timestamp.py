from datetime import datetime
import pytz

tz = pytz.timezone("Asia/Shanghai")
timestamp = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
