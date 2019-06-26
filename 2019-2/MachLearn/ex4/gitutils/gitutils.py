import subprocess

def get_sha():
    sp_obj = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
    ret = sp_obj.stdout.decode('UTF-8').strip()
    return ret
    