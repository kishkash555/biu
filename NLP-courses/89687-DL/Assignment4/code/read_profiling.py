import pstats
p = pstats.Stats('profiling.txt')
p.strip_dirs().sort_stats(-1).print_stats()