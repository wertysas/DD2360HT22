def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'cuda', '--cuda-gpu-arch=sm_75', '-std=c++11', '-nocudalib', '-Wall', '-Wextra', '-Werror' ],
  }
