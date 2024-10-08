import pandas as pd
from datetime import datetime
import pytz

utc_now = datetime.utcnow()
Beijing_tz = pytz.timezone('Asia/Beijing')
Beijing_tz = utc_now.replace(tzinfo=pytz.utc).astimezone(Beijing_tz)
# 查看结果
print(Beijing_tz)