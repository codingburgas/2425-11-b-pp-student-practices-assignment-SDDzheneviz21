from datetime import timedelta

def to_bulgarian_time(dt):
    if dt is None:
        return None
    return dt + timedelta(hours=3) 