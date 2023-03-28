# STAR
# Bisogna rifarlo andare
cd STAR
#python main_bikeNYC.py
python transfer_learning.py

# St-ResNet
cd ../ST-ResNet
#python main_bikeNYC.py
python transfer_learning.py

# # MST3D
cd ../MST3D
python transfer_learning.py
# python main_taxiBJ.py

# # Pred-CNN
cd ../Pred-CNN
python transfer_learning.py
# python main_taxiBJ.py

# # 3D-CLoST
cd ../3D-CLoST
python transfer_learning.py
# python main_taxiBJ.py

cd ../STREED-Net
python transfer_learning.py

# # ST3DNet
cd ../ST3DNet
#python prepareData.py
python transfer_learning.py
# python main_taxiBJ.py