import numpy as np
import csdl_alpha as csdl

print("csdl_alpha version:", getattr(csdl, "__version__", "unknown"))

rec = csdl.Recorder(inline=True)
rec.start()
x = csdl.Variable(value=np.array([3.0]))
y = x ** 2
try:
    d = csdl.derivative(ofs=y, wrts=x, mode="forward")
    val = d[x] if isinstance(d, dict) else d
    print("FORWARD MODE WORKS. dy/dx =", float(np.ravel(val.value)[0]), "(exact 6.0)")
except NotImplementedError as e:
    print("FORWARD MODE NOT IMPLEMENTED:", e)
except Exception as e:
    print("forward mode failed with", type(e).__name__, ":", e)
rec.stop()