import subprocess

cmds = ["/home/users/zizhaowang/hallucination/LfH/eval.py",
        "/home/users/zizhaowang/hallucination/LfD_3D/render_demo_3D.py",
        "/home/users/zizhaowang/hallucination/LfD_3D/LfD_main.py"]
for cmd in cmds:
    p = subprocess.Popen(["python", cmd])
    p.communicate()
