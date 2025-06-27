py_rssa documentation
=====================

``py_rssa`` provides a minimal Python version of some ``rssa`` functionality.

Usage example::

   from py_rssa import SSA
   import numpy as np

   x = np.sin(np.linspace(0, 2*np.pi, 50))
   s = SSA(x)
   recon = s.reconstruct([0])
   w = s.wcor()
