import datetime
import time

# t = time.localtime()
# print(t)
# current_time = time.strftime("%H:%M:%S", t)
# print(current_time)
# last_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
# print(datetime.datetime.utcnow() - datetime.timedelta(minutes=1) > last_time)
# t = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
# print(t)
t = datetime.datetime.utcnow().minute
print(t)
