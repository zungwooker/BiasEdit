import json
import time
from datetime import timedelta

class Timer():
    def __init__(self, save_path) -> None:
        self.start = None
        self.end = None
        self.records = {}
        self.save_path = save_path
        
    def save(self):
        with open(self.save_path, 'w') as file:
            json.dump(self.records, file, indent=4)
        
    def start_record(self, task_name: str):
        while task_name in self.records:
            task_name += '_new'

        self.records[task_name] = {
            'start': time.time(),
            'end': None,
            'start_localtime': time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()),
            'end_localtime': None,
            'runtime': None,
        }
        self.save()
        
    def end_record(self, task_name: str):
        if task_name not in self.records:
            print(f"[Error] Timer: {task_name} is not started.")
            return
        else:
            self.records[task_name]['end'] = time.time()
            self.records[task_name]['end_localtime'] = time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())
            runtime_seconds = int(self.records[task_name]['end'] - self.records[task_name]['start'])
            self.records[task_name]['runtime'] = str(timedelta(seconds=runtime_seconds))
        self.save()