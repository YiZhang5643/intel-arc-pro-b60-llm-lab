"""Create an MSVC import-library definition from dumpbin exports of the installed loader."""
import re
import sys
from pathlib import Path

if __name__ == '__main__':
    text = Path(sys.argv[1]).read_text(errors='replace')
    names = re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(vk\w+)\s*$',text,re.M)
    if 'vkGetInstanceProcAddr' not in names or len(names) < 50:
        raise RuntimeError('Unexpected Vulkan loader exports')
    Path(sys.argv[2]).write_text('LIBRARY vulkan-1.dll\nEXPORTS\n'+'\n'.join(names)+'\n')
    print('Vulkan loader exports:', len(names))
