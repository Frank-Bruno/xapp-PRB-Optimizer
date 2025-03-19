import csv
import numpy as np
from time import sleep
import threading

class AuxXappRl:
    def __init__(self):
        self.obs_file = "/tmp/src/obs.csv"
        self.action_file = "/tmp/src/action.csv"

    def write_obs(self, slice_1_avg_thr, slice_2_avg_thr):
        with open(self.obs_file, mode='w') as file:
            obs_write = csv.writer(file)
            obs_write.writerow([slice_1_avg_thr, slice_2_avg_thr])

    def read_action(self, time=0.0):
        action_available = False
        while not action_available:
            try:
                with open(self.action_file, mode='r') as file:
                    action_read = csv.reader(file)
                    for i, row in enumerate(action_read):
                        if i == 0:
                            action_csv = np.array([float(value) for value in row])
                            action_available = True
                            break
                if action_available:
                    with open(self.action_file, mode='w') as file:
                        file.truncate()
            except Exception:
                pass
            sleep(time)
        return action_csv

    def run(self):
        # Observation
        slice_1_avg_thr = 4
        slice_2_avg_thr = 20

        while True:
            self.write_obs(slice_1_avg_thr, slice_2_avg_thr)
            self.read_action(time=0.001)

if __name__ == "__main__":
    fake_xapp_rl = AuxXappRl()
    fake_xapp_rl.run()
