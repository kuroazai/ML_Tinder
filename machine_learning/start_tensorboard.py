import os
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', os.path.expanduser('~/.tensorboard')])
url = tb.launch()
print("TensorBoard started at %s" % url)
