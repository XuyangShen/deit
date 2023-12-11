
import os
import re

path = "./arun_log"

def parse(file, arr, id):
    res = dict()
    for a in arr:
        res[a] = 0
    res["id"] = id
    res["num_heads"] = -1
    
    with open(file) as f:
        for line in f.readlines():
            if "heads" in line:
                data = line.split()[-1]
                res["num_heads"] = data
            if "Namespace" in line:
                matchObj = re.match(r"(.*)model=(.*?),", line)
                if matchObj:
                    res["model"] = matchObj.group(2).strip("'")
                matchObj = re.match(r"(.*)batch_size=(.*?),", line)
                if matchObj:
                    res["batch_size"] = int(matchObj.group(2))
                matchObj = re.match(r"(.*), lr=(.*?),", line)
                if matchObj:
                    res["lr"] = matchObj.group(2)
                matchObj = re.match(r"(.*)clip_grad=(.*?),", line)
                if matchObj:
                    res["clip_grad"] = matchObj.group(2)
                matchObj = re.match(r"(.*)world_size=(.*?)\)", line)
                if matchObj:
                    res["gpus"] = int(matchObj.group(2).split(',')[0])
                matchObj = re.match(r"(.*)weight_decay=(.*?),", line)
                if matchObj:
                    res["weight_decay"] = float(matchObj.group(2))
                matchObj = re.match(r"(.*)warmup_epochs=(.*?),", line)
                if matchObj:
                    res["warmup_epochs"] = float(matchObj.group(2))
                break
    res["batch_size"] *= res.get("gpus", 0)
    res["acc"] = 0

    time = 0
    cnt = 0
    
    with open(file) as f:
        for line in f.readlines():
            if "params:" in line:
                data = float(line.split()[-1])
                res["params"] = data / 1000000
            if "Max accuracy:" in line:
                data = float(line.split()[-1][:-1])
                res["acc"] = max(res["acc"], data)
            if "on epoch" in line:
                data = float(line.split()[-1])
                res["epoch"] = data
            if "Epoch: " in line:
                data = line.split()[1]
                res["epoch"] = data[1:-1]
            if "Total time:" in line and "Epoch:" in line:
                data = line.split()[-5].split(":")
                cur_time = float(data[0]) * 60 + float(data[1]) + float(data[2]) / 60
                time += cur_time
                cnt += 1
            if "model:" in line:
                res["model"] = line.split()[-1]

    # if cnt == 0:
    #     return
    res["time"] = round(time / max(cnt, 1), 2)
    string = str(res[arr[0]])
    for a in arr[1:]:
        string += f",{res.get(a, -1)}"
    print(string)

##################### new
job_ids = [2355216, 2355919]

arr = ["model", "params", "acc", "epoch", "batch_size", "lr", "clip_grad", "id", "time", "weight_decay", "num_heads", "warmup_epochs"]
col = arr[0]
for a in arr[1:]:
    col += f",{a}"
print(col)

for job_id in job_ids:
    job_id = str(job_id)
    for file in os.listdir(path):
        if job_id in file:
            # print(file)
            abs_path = os.path.join(path, file)
            # nmt
            parse(abs_path, arr, job_id)
