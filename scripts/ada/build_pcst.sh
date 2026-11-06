#!/bin/bash
# pcst_fast 1.0.10 from the sdist with ftm/retrieval/pcst_fast_binding.patch (the PyPI wheel returns garbage index arrays; ftm/retrieval/pcst.py::check_binding refuses it).  usage: build_pcst.sh <scratch> <repo with the patch>
SC=$1; PATCH=${2:-$HOME/tt/ada_scripts/pcst_fast_binding.patch}; P=$SC/venv/bin/python
export UV_CACHE_DIR=$SC/uvcache TMPDIR=$SC/tmp PATH=$SC/venv/bin:$SC/bin:$PATH
set -e; mkdir -p $SC/pcst && cd $SC/pcst && rm -rf *
uv pip install -q -p $P pybind11 setuptools wheel
$P -m pip --version >/dev/null 2>&1 || uv pip install -q -p $P pip
curl -sSL -m 120 -o pcst_fast-1.0.10.tar.gz https://files.pythonhosted.org/packages/d4/fd/64b51c867bad63e6622ff97cad6230b94b19b5a61e30424cd69c8353091c/pcst_fast-1.0.10.tar.gz
echo "3b5694110ce2e004471f383267d5e4ab7fe1ba9828954e8c42560ac1e42b25e6  pcst_fast-1.0.10.tar.gz" | sha256sum -c -
tar xzf pcst_fast-1.0.10.tar.gz && cd pcst_fast-1.0.10 && patch -p1 < $PATCH
$P -m pip install -q --no-build-isolation --no-deps .
[ -d $HOME/tt/tt-f949 ] && $P -c "import sys; sys.path.insert(0, '$HOME/tt/tt-f949'); from ftm.retrieval.pcst import check_binding; check_binding(); print('pcst binding ok')"
