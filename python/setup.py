try:
    import build_info
except ImportError:
    import os
    import subprocess
    import warnings
    warnings.warn("No build information available. <CMAKE>")
    cmake_dir = os.path.join("..", "build")
    if not os.path.exists(cmake_dir):
        os.makedirs(cmake_dir)
    subprocess.call("cd %s; cmake ..; make install" % cmake_dir)

if __name___ == "__main__":
    import os
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Disutils import build_ext
    import numpy

    extra_compile_args = build_info.compile_args
    extra_compile_args.append("-Wno-enum-compare")

    setup(
        name='iNeural',
        version=build_info.version,
        description="Py bindings for iNeural.",
        author="Fatih Küçükkarakurt",
        author_email="fatihkkarakurt128@gmail.com",
        url="https://github.com/fkkarakurt/iNeural",
        license="MIT",
        ext_modules=[

        ],
        cmdclass = {"build_ext": build_ext},
    )
