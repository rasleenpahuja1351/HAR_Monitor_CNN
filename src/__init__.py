"""HAR Monitor source package (Group EIC-16).

Only ``src.inference`` is used by the deployed API; it depends on NumPy alone.
The training modules import TensorFlow / scikit-learn lazily, so importing this
package never pulls heavy libraries into the Vercel function.
"""
