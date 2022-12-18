def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'cuda', '--cuda-gpu-arch=sm_75', '-std=c++11', '-nocudalib', '-Wall', '-Wextra', '-Werror' '-I/home/tailor/code/KTH/DD2360HT22/hw_3/ex_4 /include' ],
  }
