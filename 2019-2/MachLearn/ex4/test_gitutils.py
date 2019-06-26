import sys
import gitutils.gitutils as gu

if __name__ == "__main__":
    print(list(gu.__dict__.keys()))
    print(list(gu.__dict__['__builtins__'].keys()))
    
    #print(gu.gitutils.get_sha())