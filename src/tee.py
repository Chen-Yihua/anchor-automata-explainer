# 將輸出同時輸出到終端和檔案
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if not stream.closed:
                stream.write(data)

    def flush(self):
        for stream in self.streams:
            if not stream.closed:
                stream.flush()