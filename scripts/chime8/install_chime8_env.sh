pip install espnet
git clone https://github.com/espnet/espnet.git /workspace/espnet
# choose the cupy version according to your cuda version
pip uninstall -y 'cupy-cuda118' # uninstall old one if needed
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.3.0"
pip install git+http://github.com/desh2608/gss
pip install optuna
pip install --upgrade jiwer
pip install git+https://github.com/kpu/kenlm
pip install cmake>=3.18
sh ./run_install_lm.sh ${PWD}/../../../NeMo
