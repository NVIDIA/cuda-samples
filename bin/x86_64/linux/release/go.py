import os
import subprocess
for filename in os.listdir('./'):
    if os.access(filename, os.X_OK) and os.path.isfile(filename):
        f=open("APM_%s.txt" %filename, "w")
        print("Running %s" %filename)
        p1=subprocess.Popen(
           [
               "./%s" %filename
           ],
           stdout=f,
           )
        print("Wait")
        p1.wait()
        print("Done")
        f.close()


