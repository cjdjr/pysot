

TINY_SPACE =  ['none', 
               'skip_connect', 
               'nor_conv_1x1', 
               'nor_conv_3x3', 
               'avg_pool_s1'
              ]

DARTS_SPACE = ['none',
               'max_pool_3x3',
               'avg_pool_3x3',
               'skip_connect',
               'sep_conv_3x3',
               'sep_conv_5x5',
               'dil_conv_3x3',
               'dil_conv_5x5'
              ]

_SEARCH_SPACE = {
  'GDAS-Tiny-SEARCH-SPACE': TINY_SPACE,
  'DARTS-SEARCH-SPACE':     DARTS_SPACE,
}

def build_search_space(cfg):
  seach_space = _SEARCH_SPACE[cfg.SEARCH.SEARCH_SPACE]
  return seach_space
  