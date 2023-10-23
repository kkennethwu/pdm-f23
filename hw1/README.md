# pdm-f23-hw1

NYCU Perception and Decision Making 2023 Fall

Spec: [Google Docs](https://docs.google.com/document/d/1whwLunr64Q5aqhjNhRfl7udZV4_0wfdZ/edit?usp=sharing&ouid=111927449078729907735&rtpof=true&sd=true)

## Preparation
In your original dpm-f23 directory, `git pull` to get new `hw1` directory.

As for replica dataset, you can use the same one in `hw0`.

## Run my code 
```
# bev 
python bev.py

# reconstruction
# k is floor number
python load.py -f {k}
python reconstruction.py -f {k} -v open3d
python reconstruction.py -f {k} -v my_icp

```