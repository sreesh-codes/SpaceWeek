"""Run 01_gee_sentinel.py, casting images to float32 before direct download.
S1_GRD bands are float64, which pushes each clip just past EE's 48 MB
direct-download cap; float32 halves the size with no meaningful precision loss."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gee", "01_gee_sentinel.py")
gee = importlib.util.module_from_spec(spec); spec.loader.exec_module(gee)
_dl = gee.download
gee.download = lambda image, geom, path: _dl(
    image if path.split("_")[-2] == "s2" else image.toFloat(), geom, path)
sys.argv[0] = "01_gee_sentinel.py"
gee.main()
