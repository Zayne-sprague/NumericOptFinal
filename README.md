Project Website: https://hyperplane-lab.github.io/Generative-3D-Part-Assembly/
Base code is at https://github.com/hyperplane-lab/Generative-3D-Part-Assembly

What we should do for Q2
- 

1.) Put their code into /src/{their repo} 
    
Delete their .git file and check it into our repo

2.) Get their pretrained model / dataset

Dataset is http://download.cs.stanford.edu/orion/genpartass/prepare_data.zip

Pretrained model is http://download.cs.stanford.edu/orion/genpartass/checkpoints.zip 
put the zip in the exps file not in the prep_data area 

when unzipped it should be at /exps/prepare_data and then a bunch of npy files
(download using firefox if using linux for some reason chromium download is corrupted)

run 
    conda env create -f environment.yaml
    . activate PartAssembly
    cd exps/utils/cd
    python setup.py build

if it breaks you may need to run python setup.py install
make sure cudatoolkit is installed and the PATH is set properly

Might install some other things like scikit and scikitlearn as well as ipdb in the venv

their github says dynamic_graph_learning that's a typo

to train
 cd exps/Our_Method-dynamic_graph_learning/scripts/
    ./train_dynamic.sh

to test
  cd exps/Our_Method-dynamic_graph_learning/scripts/
    ./test_dynamic.sh

replace Our_Method-dynamic_graph_learning with the folder so Baseline-LSTM etc.

3.) Run their experiments (B-LSTM, B-Global, and their real one)

B-Complement would be cool, but might be hard to do

4.) Put any custom code in /src/ outside of their folder so we can separate their code from ours.
