import json


# filename = "raw/avail-stats-aws-2023-08-03T12_35_08.json" # 1975
filename = "raw/avail-stats-aws-2023-08-07T16_53_40.json"

zone2avails = {}

for line in open(filename):
    data = json.loads(line)
    zone2poolsize = {}
    for ins in data["instances"]:
        zone = ins["zone"]
        avail = len(ins["ins_ids"])
        if zone not in zone2poolsize:
            zone2poolsize[zone] = 0
        zone2poolsize[zone] += avail

    for zone, poolsize in zone2poolsize.items():
        if zone not in zone2avails:
            zone2avails[zone] = []
        zone2avails[zone].append(poolsize)

print(zone2avails.keys())

for zone, avails in zone2avails.items():
    out_json = {}
    out_json["metadata"] = {"gap_seconds": 300}
    # out_json["data"] = avails[1975:]
    out_json["data"] = avails[:3156]
    print(len(out_json["data"]))

    with open(f"{zone}_v100_1.json", "w") as f:
        json.dump(out_json, f)
