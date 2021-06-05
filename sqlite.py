import datetime
import sqlite3
import entry



def insert_variable_into_table(camera_time, people_count):
    e = entry.Entry(camera_time, people_count)
    # establish connection with
    connection = sqlite3.connect('days.db')
    cursor = connection.cursor()
    print("Connected to SQLite3.")

    # Insert Python variables into entry table
    sqlite_insert_param = """ INSERT INTO entry (camera_time, weekday, people_count) VALUES (?, ?, ?);"""
    data_tuple = (e.camera_time, e.weekday, e.people_count)
    cursor.execute(sqlite_insert_param, data_tuple)
    connection.commit()
    print("Variables inserted successfully into database.")

    cursor.close()
    connection.close()
    print("SQLite connection closed.")


connection = sqlite3.connect('days.db')
cursor = connection.cursor()
print("Connected to SQLite3.")
cursor.execute("SELECT * FROM entry WHERE weekday = 2")
# print(cursor.fetchall())
current_day_entries = [entry.Entry(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f'), row[2])
                       for row in cursor.fetchall()]

# for row in cursor.fetchall():
#     e = entry.Entry(datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f'), row[2])
#     print(e.camera_time)
#     print(e.weekday)
#     print(e.people_count)

# insert_variable_into_table(datetime.datetime.utcnow(), 5)
# insert_variable_into_table(datetime.datetime.utcnow() - datetime.timedelta(minutes=120), 5)
# insert_variable_into_table(datetime.datetime.utcnow() - datetime.timedelta(minutes=60), 1)
# insert_variable_into_table(datetime.datetime.utcnow() - datetime.timedelta(minutes=45), 5)
# insert_variable_into_table(datetime.datetime.utcnow() - datetime.timedelta(minutes=30), 3)
# insert_variable_into_table(datetime.datetime.utcnow() - datetime.timedelta(minutes=80), 5)

# sql = ''' CREATE TABLE entry(camera_time TIMESTAMP,
#                                 weekday INT,
#                                 people_count INT)'''
#
# cursor.execute(sql)
# connection.commit()
cursor.close()
connection.close()
print("SQLite connection closed.")